# Copyright (c) 2024 Alibaba Inc (authors: Xiang Lyu, Liu Yue)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import argparse
import numpy as np
import torch
import torchaudio
import random
import librosa
import base64
import io
from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append('{}/third_party/Matcha-TTS'.format(ROOT_DIR))
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav, logging
from cosyvoice.utils.common import set_all_random_seed

# Configure file logging
import logging as std_logging # To access FileHandler, Formatter, etc., as 'logging' is already in use.
log_file_path = os.path.join(ROOT_DIR, "api_service.log") # ROOT_DIR is defined above
file_handler = std_logging.FileHandler(log_file_path, encoding='utf-8')
file_handler.setLevel(std_logging.INFO) # Set level for this specific handler
formatter = std_logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s [%(filename)s:%(lineno)d]')
file_handler.setFormatter(formatter)

# Add the handler to the logger instance imported from cosyvoice.utils.file_utils
# This assumes 'logging' is a logger object (e.g., from logging.getLogger())
if hasattr(logging, 'addHandler'):
    logging.addHandler(file_handler)
    # Ensure the logger itself is at least INFO level to pass messages to the handler.
    # If current level is NOTSET (0) or higher (less verbose) than INFO, set it to INFO.
    if hasattr(logging, 'setLevel') and hasattr(logging, 'level'):
        if logging.level == 0 or logging.level > std_logging.INFO: # logging.NOTSET is 0
            logging.setLevel(std_logging.INFO)
    elif hasattr(logging, 'setLevel'): # If it has setLevel but not level (unlikely for std logger)
        logging.setLevel(std_logging.INFO) # Set it just in case
else:
    # Fallback: If 'logging' is not a standard logger object with addHandler,
    # try to configure the root logger. This might happen if 'logging' is just the module.
    std_logger = std_logging.getLogger()
    std_logger.addHandler(file_handler)
    if std_logger.level == 0 or std_logger.level > std_logging.INFO:
        std_logger.setLevel(std_logging.INFO)
    # Use the original logging object for this message if possible, otherwise std_logging
    if hasattr(logging, 'info'):
        logging.info("Fallback: Configured root logger for file output as 'cosyvoice.utils.file_utils.logging' did not have addHandler.")
    else:
        std_logging.info("Fallback: Configured root logger for file output as 'cosyvoice.utils.file_utils.logging' did not have addHandler.")


# ��������ģ��
class SftRequest(BaseModel):
    tts_text: str
    spk_id: str
    stream: bool = False
    speed: float = 1.0
    seed: int = 0

class ZeroShotRequest(BaseModel):
    tts_text: str
    prompt_text: str
    prompt_audio_base64: Optional[str] = None
    stream: bool = False
    speed: float = 1.0
    seed: int = 0

class CrossLingualRequest(BaseModel):
    tts_text: str
    prompt_audio_base64: Optional[str] = None
    stream: bool = False
    speed: float = 1.0
    seed: int = 0

class InstructRequest(BaseModel):
    tts_text: str
    spk_id: str
    instruct_text: str
    stream: bool = False
    speed: float = 1.0
    seed: int = 0

class SpeakerDetail(BaseModel):
    spk_id: str
    spk_name: str

class AvailableSpksResponse(BaseModel):
    speakers: List[SpeakerDetail]

class SpeakerPromptSaveRequest(BaseModel):
    spk_id: str = Field(..., description="Speaker ID to save the prompt under.")
    prompt_text: str = Field(..., description="Text accompanying the prompt audio.")
    prompt_audio_base64: str = Field(..., description="Base64 encoded prompt audio data.")
    # spk_name: Optional[str] = Field(None, description="Optional display name for the speaker.")

# ���ڴ洢�Զ�����ɫ����
SPEAKER_NAMES_FILE = "speaker_names.json"
SPEAKER_NAMES: Dict[str, str] = {}

def load_speaker_names():
    global SPEAKER_NAMES
    if os.path.exists(SPEAKER_NAMES_FILE):
        try:
            with open(SPEAKER_NAMES_FILE, 'r', encoding='utf-8') as f:
                SPEAKER_NAMES = json.load(f)
            logging.info(f"�Զ�����ɫ�����Ѵ� {SPEAKER_NAMES_FILE} ���ء�")
        except Exception as e:
            logging.error(f"�����Զ�����ɫ����ʧ��: {e}")
    else:
        logging.info(f"{SPEAKER_NAMES_FILE} δ�ҵ�����ʹ�ÿյ���ɫ�����б���")

def save_speaker_names():
    global SPEAKER_NAMES
    try:
        with open(SPEAKER_NAMES_FILE, 'w', encoding='utf-8') as f:
            json.dump(SPEAKER_NAMES, f, ensure_ascii=False, indent=4)
        logging.info(f"�Զ�����ɫ�����ѱ��浽 {SPEAKER_NAMES_FILE}��")
    except Exception as e:
        logging.error(f"�����Զ�����ɫ����ʧ��: {e}")



max_val = 0.8
prompt_sr = 16000

def postprocess(speech, top_db=60, hop_length=220, win_length=440):
    speech, _ = librosa.effects.trim(
        speech, top_db=top_db,
        frame_length=win_length,
        hop_length=hop_length
    )
    if speech.abs().max() > max_val:
        speech = speech / speech.abs().max() * max_val
    speech = torch.concat([speech, torch.zeros(1, int(cosyvoice.sample_rate * 0.2))], dim=1)
    return speech

def decode_audio(base64_audio):
    """��Base64�ַ���������Ƶ����"""
    if not base64_audio:
        return None
    
    try:
        audio_bytes = base64.b64decode(base64_audio)
        # ������ʱ�ļ��Ա�torchaudio���Զ�ȡ
        temp_file = io.BytesIO(audio_bytes)
        temp_file.name = "temp.wav"  # ΪBytesIO����������������
        return temp_file
    except Exception as e:
        logging.error(f"������Ƶʧ��: {str(e)}")
        raise HTTPException(status_code=400, detail=f"��Ƶ����ʧ��: {str(e)}")

def create_app():
    app = FastAPI(title="CosyVoice API", description="�����ϳ�API����")
    
    # ����CORS�м��
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.get("/")
    def read_root():
        return {"message": "��ӭʹ��CosyVoice API����", "version": "1.0.0"}
    
    @app.get("/available_spks", response_model=AvailableSpksResponse)
    def get_available_spks():
        """��ȡ���õ�Ԥѵ����ɫ�б��������Զ�����ɫ��"""
        spk_ids = cosyvoice.list_available_spks() # ��ͨ������Ԥ�����ͨ��add_zero_shot_spk���ӵ�
        speakers_details = []
        for spk_id in spk_ids:
            # ���ȴ�SPEAKER_NAMES��ȡ�Զ������ƣ�����Ĭ��Ϊspk_id
            spk_name = SPEAKER_NAMES.get(spk_id, spk_id)
            speakers_details.append(SpeakerDetail(spk_id=spk_id, spk_name=spk_name))
        
        # ���ڽ�������SPEAKER_NAMES�е�����δ��cosyvoice�ڲ��б��ģ������ϲ�Ӧ������������߼���ȷ��
        # ��Ϊ�������ԣ����Կ��Ǻϲ���������ǰcosyvoice.list_available_spks()ӦΪȨ����Դ
        return {"speakers": speakers_details}

    @app.post("/tts/sft")
    async def tts_sft(request: SftRequest):
        """ʹ��Ԥѵ����ɫ�ϳ�����"""
        if request.seed > 0:
            set_all_random_seed(request.seed)
        else:
            set_all_random_seed(random.randint(1, 100000000))
        
        if request.stream:
            # ��ʽ��Ӧ
            def generate():
                for i in cosyvoice.inference_sft(request.tts_text, request.spk_id, stream=request.stream, speed=request.speed):
                    # ��numpy����ת��Ϊwav��ʽ���ֽ�
                    audio_data = i['tts_speech'].numpy().flatten()
                    wav_bytes = convert_to_wav(audio_data, cosyvoice.sample_rate)
                    yield wav_bytes
            
            return StreamingResponse(generate(), media_type="audio/wav")
        else:
            # ����ʽ��Ӧ���ռ�������ƵƬ��
            audio_segments = []
            for i in cosyvoice.inference_sft(request.tts_text, request.spk_id, stream=request.stream, speed=request.speed):
                audio_segments.append(i['tts_speech'].numpy().flatten())
            
            # �ϲ�������ƵƬ��
            if audio_segments:
                combined_audio = np.concatenate(audio_segments)
                wav_bytes = convert_to_wav(combined_audio, cosyvoice.sample_rate)
                return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
            else:
                raise HTTPException(status_code=500, detail="��Ƶ����ʧ��")
    
    @app.post("/tts/zero_shot")
    async def tts_zero_shot(request: ZeroShotRequest):
        """ʹ��3s���ٸ���ģʽ�ϳ�����"""
        if not request.prompt_audio_base64:
            raise HTTPException(status_code=400, detail="prompt��Ƶ����Ϊ��")
        
        if not request.prompt_text:
            raise HTTPException(status_code=400, detail="prompt�ı�����Ϊ��")
        
        # ����Base64��Ƶ
        prompt_wav_file = decode_audio(request.prompt_audio_base64)
        if not prompt_wav_file:
            raise HTTPException(status_code=400, detail="prompt��Ƶ����ʧ��")
        
        try:
            # ���ز�������Ƶ
            prompt_speech_16k = postprocess(load_wav(prompt_wav_file, prompt_sr))
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            if request.stream:
                # ��ʽ��Ӧ
                def generate():
                    for i in cosyvoice.inference_zero_shot(request.tts_text, request.prompt_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                        audio_data = i['tts_speech'].numpy().flatten()
                        wav_bytes = convert_to_wav(audio_data, cosyvoice.sample_rate)
                        yield wav_bytes
                
                return StreamingResponse(generate(), media_type="audio/wav")
            else:
                # ����ʽ��Ӧ
                audio_segments = []
                for i in cosyvoice.inference_zero_shot(request.tts_text, request.prompt_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                    audio_segments.append(i['tts_speech'].numpy().flatten())
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    wav_bytes = convert_to_wav(combined_audio, cosyvoice.sample_rate)
                    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
                else:
                    raise HTTPException(status_code=500, detail="��Ƶ����ʧ��")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"����ʧ��: {str(e)}")
    
    @app.post("/tts/cross_lingual")
    async def tts_cross_lingual(request: CrossLingualRequest):
        """ʹ�ÿ����ָ���ģʽ�ϳ�����"""
        if not request.prompt_audio_base64:
            raise HTTPException(status_code=400, detail="prompt��Ƶ����Ϊ��")
        
        # ����Base64��Ƶ
        prompt_wav_file = decode_audio(request.prompt_audio_base64)
        if not prompt_wav_file:
            raise HTTPException(status_code=400, detail="prompt��Ƶ����ʧ��")
        
        try:
            # ���ģ���Ƿ�֧�ֿ����ָ���
            if cosyvoice.instruct is True:
                raise HTTPException(status_code=400, detail=f"��ǰģ�Ͳ�֧�ֿ����ָ���ģʽ����ʹ��CosyVoice-300Mģ��")
            
            # ���ز�������Ƶ
            prompt_speech_16k = postprocess(load_wav(prompt_wav_file, prompt_sr))
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            if request.stream:
                # ��ʽ��Ӧ
                def generate():
                    for i in cosyvoice.inference_cross_lingual(request.tts_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                        audio_data = i['tts_speech'].numpy().flatten()
                        wav_bytes = convert_to_wav(audio_data, cosyvoice.sample_rate)
                        yield wav_bytes
                
                return StreamingResponse(generate(), media_type="audio/wav")
            else:
                # ����ʽ��Ӧ
                audio_segments = []
                for i in cosyvoice.inference_cross_lingual(request.tts_text, prompt_speech_16k, stream=request.stream, speed=request.speed):
                    audio_segments.append(i['tts_speech'].numpy().flatten())
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    wav_bytes = convert_to_wav(combined_audio, cosyvoice.sample_rate)
                    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
                else:
                    raise HTTPException(status_code=500, detail="��Ƶ����ʧ��")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"����ʧ��: {str(e)}")
    
    @app.post("/tts/instruct")
    async def tts_instruct(request: InstructRequest):
        """ʹ����Ȼ���Կ���ģʽ�ϳ�����"""
        try:
            # ���ģ���Ƿ�֧����Ȼ���Կ���
            if cosyvoice.instruct is False:
                raise HTTPException(status_code=400, detail=f"��ǰģ�Ͳ�֧����Ȼ���Կ���ģʽ����ʹ��CosyVoice-300M-Instructģ��")
            
            if not request.instruct_text:
                raise HTTPException(status_code=400, detail="instruct�ı�����Ϊ��")
            
            if request.seed > 0:
                set_all_random_seed(request.seed)
            else:
                set_all_random_seed(random.randint(1, 100000000))
            
            if request.stream:
                # ��ʽ��Ӧ
                def generate():
                    for i in cosyvoice.inference_instruct(request.tts_text, request.spk_id, request.instruct_text, stream=request.stream, speed=request.speed):
                        audio_data = i['tts_speech'].numpy().flatten()
                        wav_bytes = convert_to_wav(audio_data, cosyvoice.sample_rate)
                        yield wav_bytes
                
                return StreamingResponse(generate(), media_type="audio/wav")
            else:
                # ����ʽ��Ӧ
                audio_segments = []
                for i in cosyvoice.inference_instruct(request.tts_text, request.spk_id, request.instruct_text, stream=request.stream, speed=request.speed):
                    audio_segments.append(i['tts_speech'].numpy().flatten())
                
                if audio_segments:
                    combined_audio = np.concatenate(audio_segments)
                    wav_bytes = convert_to_wav(combined_audio, cosyvoice.sample_rate)
                    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
                else:
                    raise HTTPException(status_code=500, detail="��Ƶ����ʧ��")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"����ʧ��: {str(e)}")
    
    # �ļ��ϴ��ӿڣ������ϴ�prompt��Ƶ
    @app.post("/speaker_prompt/save")
    async def save_speaker_prompt(prompt_text: str = Form(...), 
                              spk_id: str = Form(...), 
                              spk_name: Optional[str] = Form(None), # ������ɫ���Ʋ���
                              prompt_audio: UploadFile = File(...)):
        """�����û��ϴ�����ɫprompt"""
        if not prompt_audio:
            raise HTTPException(status_code=400, detail="prompt_audio ��Ƶ�ļ�����Ϊ��")
    
        if not prompt_text:
            raise HTTPException(status_code=400, detail="prompt_text ����Ϊ��")

        if not spk_id:
            raise HTTPException(status_code=400, detail="spk_id ����Ϊ��")

        # �� UploadFile �����ȡ��Ƶ
        # ע�⣺load_wav ��Ҫ�ܹ����� UploadFile.file������һ�� SpooledTemporaryFile
        # �����Ƚ� UploadFile ���浽��ʱ�ļ��ٶ�ȡ
        # Ϊ�������������� load_wav ����ֱ�Ӵ��� file-like object
        # ������У���Ҫ�� await prompt_audio.read() Ȼ���� io.BytesIO ��װ
        prompt_wav_file = prompt_audio.file

        try:
            # ���ز�������Ƶ
            prompt_speech_16k = postprocess(load_wav(prompt_wav_file, prompt_sr))

            # Use frontend_zero_shot to get all necessary features including embeddings
            # Pass tts_text='' and zero_shot_spk_id='' to ensure fresh extraction
            model_input_data = cosyvoice.frontend.frontend_zero_shot(
                tts_text='',
                prompt_text=prompt_text,
                prompt_speech_16k=prompt_speech_16k,
                resample_rate=cosyvoice.sample_rate,
                zero_shot_spk_id=''
            )

            # Remove fields related to tts_text as they are not part of a speaker prompt
            if 'text' in model_input_data:
                del model_input_data['text']
            if 'text_len' in model_input_data:
                del model_input_data['text_len']

            # Add the 'embedding' key directly for SFT compatibility,
            # using the llm_embedding (which is the speaker embedding).
            if 'llm_embedding' in model_input_data:
                model_input_data['embedding'] = model_input_data['llm_embedding']
            else:
                # Fallback or error if llm_embedding is somehow missing
                logging.error("llm_embedding not found in frontend_zero_shot output. Cannot save speaker for SFT.")
                raise HTTPException(status_code=500, detail="��ɫ������ȡʧ�ܣ�ȱ��llm_embedding")

            # Store the processed information in spk2info
            cosyvoice.frontend.spk2info[spk_id] = model_input_data

            # Persist the updated spk2info
            cosyvoice.save_spkinfo()

            # ���������Զ�����ɫ����
            actual_spk_name = spk_name if spk_name and spk_name.strip() else spk_id
            SPEAKER_NAMES[spk_id] = actual_spk_name
            save_speaker_names() # �����Զ�����ɫ���Ƶ�JSON�ļ�

            return JSONResponse(content={"status": "success", "message": f"��ɫ '{actual_spk_name}' (ID: {spk_id}) ����ɹ�", "spk_id": spk_id, "spk_name": actual_spk_name}, status_code=200)
        except Exception as e:
            # print(e)
            logging.error(f"������ɫ {spk_id} ʧ�� (raw): {e}") # Log with repr for more detail
            # Sanitize the error message for the HTTP response
            detail_message = f"������ɫ {spk_id} ʧ��:"
            raise HTTPException(status_code=500, detail=detail_message)

    # �ļ��ϴ��ӿڣ������ϴ�prompt��Ƶ
    @app.post("/upload_audio")
    async def upload_audio(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            # ���ļ����ݱ���Ϊbase64
            base64_audio = base64.b64encode(contents).decode('utf-8')
            return {"filename": file.filename, "audio_base64": base64_audio}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"�ϴ�ʧ��: {str(e)}")
    
    return app

def convert_to_wav(audio_data, sample_rate):
    """��numpy����ת��ΪWAV��ʽ���ֽ�"""
    # ��float32ת��Ϊint16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # ����WAV�ļ�ͷ
    buffer = io.BytesIO()
    with io.BytesIO() as wav_file:
        # ����WAV�ļ�
        wav_writer = wave.open(wav_file, 'wb')
        wav_writer.setnchannels(1)  # ������
        wav_writer.setsampwidth(2)  # 16λ
        wav_writer.setframerate(sample_rate)
        wav_writer.writeframes(audio_int16.tobytes())
        wav_writer.close()
        
        # ��ȡWAV�ļ����ֽ�
        wav_file.seek(0)
        wav_bytes = wav_file.read()
    
    return wav_bytes

import wave # ȷ��wave�ڶ�������

def main():
    import uvicorn
    # import wave # ���Ƶ�����
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--port',
                        type=int,
                        default=8000)
    parser.add_argument('--model_dir',
                        type=str,
                        default='pretrained_models/CosyVoice2-0.5B',
                        help='local path or modelscope repo id')
    parser.add_argument('--vllm',
                        action='store_true',
                        default=False,
                        help='use vllm for inference')
    args = parser.parse_args()
    
    global cosyvoice, default_data
    try:
        cosyvoice = CosyVoice2(args.model_dir,
                               load_jit=False,
                               load_trt=False,
                               load_vllm=args.vllm,
                               fp16=True)
    except Exception:
        raise TypeError('no valid model_type!')
    
    default_data = np.zeros(cosyvoice.sample_rate)

    load_speaker_names() # Ӧ������ʱ�����Զ�����ɫ����
    
    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=args.port)

if __name__ == '__main__':
    main()