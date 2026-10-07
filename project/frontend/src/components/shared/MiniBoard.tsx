import Image from "next/image";

const pieces: Record<number,string> = {
  0:"black-xe.png",1:"black-ma.png",2:"black-tuongj.png",3:"black-si.png",4:"black-tuong.png",5:"black-si.png",6:"black-tuongj.png",7:"black-ma.png",8:"black-xe.png",
  19:"black-phao.png",25:"black-phao.png",27:"black-tot.png",29:"black-tot.png",31:"black-tot.png",33:"black-tot.png",35:"black-tot.png",
  54:"red-tot.png",56:"red-tot.png",58:"red-tot.png",60:"red-tot.png",62:"red-tot.png",64:"red-phao.png",70:"red-phao.png",
  81:"red-xe.png",82:"red-ma.png",83:"red-tuongj.png",84:"red-si.png",85:"red-tuong.png",86:"red-si.png",87:"red-tuongj.png",88:"red-ma.png",89:"red-xe.png"
};

export function MiniBoard() {
  return (
    <div className="board" aria-label="Xiangqi board preview">
      {Array.from({length:90},(_,i)=>(
        <div className="board-cell" key={i}>
          {pieces[i] ? <Image src={`/assets/pieces/${pieces[i]}`} alt="" width={42} height={42} /> : null}
        </div>
      ))}
    </div>
  );
}
