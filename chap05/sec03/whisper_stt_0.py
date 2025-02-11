import os # ①
import torch # ①
import pandas as pd # ①
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline 

os.environ["PATH"] += os.pathsep + r"C:\github\gpt_agent_2025_easyspub\ffmpeg-2025-02-10-full_build\bin" # 자신이 설치한 위치로 경로 수정

def whisper_stt(
    audio_file_path: str,      
    output_file_path: str = "./output.csv"
):  # ②
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_id = "openai/whisper-large-v3-turbo"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True, 
        use_safetensors=True
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_id)

    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch_dtype,
        device=device,
        return_timestamps=True,  # 청크별로 타임스탬프를 반환
        chunk_length_s=10,  # 입력 오디오를 10초씩 나누기
        stride_length_s=2,  # 2초씩 겹치도록 청크 나누기
    )

    result = pipe(audio_file_path)
    df = whisper_to_dataframe(result, output_file_path)  # ③ 아래에 선언되어 있음 

    return result, df


def whisper_to_dataframe(result, output_file_path): # ③
    start_end_text = []

    for chunk in result["chunks"]:
        start = chunk["timestamp"][0]
        end = chunk["timestamp"][1]
        text = chunk["text"].strip()
        start_end_text.append([start, end, text])
        df = pd.DataFrame(start_end_text, columns=["start", "end", "text"])
        df.to_csv(output_file_path, index=False, sep="|")
    
    return df


if __name__ == "__main__":
    result, df = whisper_stt(
        "./chap05/audio/싼기타_비싼기타.mp3", 
        "./chap05/audio/싼기타_비싼기타.csv", 
    )

    print(df)
