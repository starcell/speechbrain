import wave
import librosa

def verify_audio_file(audio_file_path:str) -> bool:
    """ audio file 검사, 비정상적 wav 파일 검출
    """
    try:
        wave.open(audio_file_path, mode='rb').close()
        return True
    except:
        # logger.info(f'wrong file : {audio_file_path}')
        print(f'wrong file : {audio_file_path}')
        return False


def check_sample_rate(audio_file_path:str, sample_rate:int=16000) -> int:
    """ 오디오파일의 샘플레이트가 주어진 값과 같은지 확인
    """
    # wrong_audio_dic = {"file":"", "file_open":"", "sample_rate":""}
    wrong_audio_list = [] # [wrong_audio_dic]
    
    try:
        sr = librosa.get_samplerate(audio_file_path)
        if  sr != sample_rate:   ### sample rate(frame rate) 검사, 16000이 아닌 파일 검출
            wrong_audio_list.append({"file":audio_file_path, "file_openr":"ok", "sample_rate":sr})
    except FileNotFoundError:
        wrong_audio_list.append({"file":audio_file_path, "file_openr":"FileNotFound", "sample_rate":0})
    except Exception:
        wrong_audio_list.append({"file":audio_file_path, "file_openr":Exception, "sample_rate":0})

    return wrong_audio_list
