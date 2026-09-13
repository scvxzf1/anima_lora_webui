import { useState } from "react";
import { RefreshCw } from "lucide-react";

export function CaptionPreviewImage({ src, name }: { src: string; name: string }) {
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  return <div className="caption-image-preview">
    {failed || !src ? <div className="caption-image-error" role="alert">
      <p>无法读取图片：{name}</p>
      {src ? <button type="button" onClick={() => { setFailed(false); setAttempt(attempt + 1); }}><RefreshCw size={15} />重试图片</button> : <p>未保存可用图片地址</p>}
    </div> : <img key={attempt} className="caption-original" src={src} alt={name} onError={() => setFailed(true)} />}
  </div>;
}
