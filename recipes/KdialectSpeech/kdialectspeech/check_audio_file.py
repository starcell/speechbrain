import wave
import librosa

def check_audio_file(audio_file_path:str, sample_rate:int=16000):
    """ audio file 검사, 비정상적 wav 파일 검출
        sample rate 검사
    """
    # 1. wav file open 시도하고 정상적으로 open되는 파일만 샘플레이트 검사
    try:
        wave.open(audio_file_path, mode='rb').close()
        sr = librosa.get_samplerate(audio_file_path)

        # 2. wav file open이 성공하면 샘플레이트 검사
        if  sr != sample_rate:   ### sample rate(frame rate) 검사, 16000이 아닌 파일 검출
            return {"file":audio_file_path, "file_open":"ok", "sample_rate":sr}
        
        return None

    except FileNotFoundError:
        return {"file":audio_file_path, "file_open":"FileNotFound", "sample_rate":0}    

    except Exception:
        return {"file":audio_file_path, "file_open":Exception, "sample_rate":0}