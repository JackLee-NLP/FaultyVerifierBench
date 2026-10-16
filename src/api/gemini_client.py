from __future__ import annotations
import os,random,time
from typing import Optional
from dotenv import load_dotenv
from google import genai
from google.genai import types
from src.api.base_client import BaseLLMClient
load_dotenv()

class GeminiClient(BaseLLMClient):
    LEVELS={"low","medium","high"}
    def __init__(self,*args,thinking_level:str="high",include_thoughts:bool=False,**kwargs):
        super().__init__(*args,**kwargs);self.thinking_level=str(thinking_level).strip().lower();self.include_thoughts=bool(include_thoughts)
        if self.thinking_level not in self.LEVELS:raise ValueError("Gemini thinking_level must be low, medium, or high")
    def _key(self):
        key=(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
        if not key:raise RuntimeError("Neither GEMINI_API_KEY nor GOOGLE_API_KEY is set")
        return key
    def _parts(self,chunk):
        candidates=getattr(chunk,"candidates",None) or []
        if not candidates:return []
        return getattr(getattr(candidates[0],"content",None),"parts",None) or []
    def _extract(self,stream):
        answers=[];chunks=thoughts=thought_chars=0;start=time.time();last=start
        for chunk in stream:
            chunks+=1
            for part in self._parts(chunk):
                text=getattr(part,"text",None)
                if not text:continue
                if bool(getattr(part,"thought",False)):thoughts+=1;thought_chars+=len(str(text))
                else:answers.append(str(text))
            if time.time()-last>=10:
                print(f"Streaming progress | chunks={chunks} | thought_chunks={thoughts} | thought_characters={thought_chars} | answer_characters={sum(map(len,answers))}",flush=True);last=time.time()
        output="".join(answers).strip()
        if not output:raise RuntimeError("Gemini stream contained no final answer text")
        return output
    def generate(self,system_prompt:str,user_prompt:str)->str:
        last:Optional[Exception]=None
        for attempt in range(1,self.max_retries+1):
            client=None;stream=None;print(f"Streaming attempt {attempt}/{self.max_retries} | model={self.model_name} | max_output_tokens={self.max_output_tokens} | thinking_level={self.thinking_level} | include_thoughts={self.include_thoughts}",flush=True)
            try:
                client=genai.Client(api_key=self._key(),http_options=types.HttpOptions(api_version="v1beta",timeout=int(max(float(self.connect_timeout),float(self.read_timeout))*1000)))
                config=types.GenerateContentConfig(system_instruction=system_prompt,temperature=float(self.temperature),max_output_tokens=self.max_output_tokens,thinking_config=types.ThinkingConfig(thinking_level=self.thinking_level,include_thoughts=self.include_thoughts))
                stream=client.models.generate_content_stream(model=self.model_name,contents=user_prompt,config=config);return self._extract(stream)
            except Exception as e:
                last=e;print(f"Request failed | attempt={attempt}/{self.max_retries} | error={e!r}",flush=True)
                if any(x in str(e).lower() for x in ("400","401","403","404","invalid api key","unauthenticated","invalid argument")):raise RuntimeError(f"Gemini non-retryable error: {e}") from e
                if attempt<self.max_retries:time.sleep(float(self.retry_base_delay)*(2**(attempt-1))+random.uniform(0,float(self.retry_base_delay)))
            finally:
                for obj in (stream,client):
                    if obj is not None:
                        try:
                            close=getattr(obj,"close",None)
                            if callable(close):close()
                        except Exception:pass
        raise RuntimeError(f"Gemini failed after {self.max_retries} attempts: {last!r}")
