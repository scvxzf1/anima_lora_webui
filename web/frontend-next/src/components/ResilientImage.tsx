import { useState, type ImgHTMLAttributes } from "react";
import { ImageOff, RefreshCw } from "lucide-react";
import "./ResilientImage.css";

type Props = ImgHTMLAttributes<HTMLImageElement> & { retry?: boolean };

export function ResilientImage(props: Props) {
  return <ImageContent key={props.src} {...props} />;
}

function ImageContent({ src, alt = "", className = "", retry = false, ...props }: Props) {
  const [failed, setFailed] = useState(false);
  const [attempt, setAttempt] = useState(0);
  if (src && !failed) return <img {...props} key={attempt} className={className} src={src} alt={alt} onError={() => setFailed(true)} />;
  return <span className={`image-fallback ${className}`}>
    <span role="img" aria-label={`无法读取图片${alt ? `：${alt}` : ""}`} title="无法读取图片"><ImageOff size={20} /></span>
    {retry && <><span>无法读取图片{alt ? `：${alt}` : ""}</span>
      {src ? <button type="button" onClick={() => { setFailed(false); setAttempt(attempt + 1); }}><RefreshCw size={15} />重试图片</button>
        : <span>未保存可用图片地址</span>}</>}
  </span>;
}
