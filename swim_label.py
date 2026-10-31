"""
swim_label.py  -  instructor labelling tool (2=Well 1=Marginal 0=NotFloating a=Assisted u=Unknown s=skip q=quit)
Usage: python swim_label.py features/sideFront_features.csv videos/sideFront.mp4 --out labelled.csv
"""
import argparse, cv2, pandas as pd, numpy as np
def main(feat_csv, video, out):
    df=pd.read_csv(feat_csv); cap=cv2.VideoCapture(video); fps=cap.get(cv2.CAP_PROP_FPS) or 25
    if "float_label" not in df: df["float_label"]=np.nan
    m={"2":"Well","1":"Marginal","0":"NotFloating","a":"Assisted","u":"Unknown"}
    for i,w in df.iterrows():
        if not pd.isna(df.at[i,"float_label"]): continue
        mid_t=(w["start_s"]+w["end_s"])/2; cap.set(cv2.CAP_PROP_POS_MSEC,mid_t*1000); ok,fr=cap.read()
        if ok:
            cv2.putText(fr,f"{w['start_s']:.0f}-{w['end_s']:.0f}s 2=Well 1=Marg 0=No a=Assist u=Unk s=skip q=quit",
                        (20,40),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,255),2)
            cv2.imshow("label",fr)
        key=chr(cv2.waitKey(0)&0xFF)
        if key=="q": break
        if key=="s": continue
        if key in m: df.at[i,"float_label"]=m[key]
    cap.release(); cv2.destroyAllWindows(); df.to_csv(out,index=False)
    print(f"saved {out}; labelled {df['float_label'].notna().sum()} windows")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features"); ap.add_argument("video"); ap.add_argument("--out",default="labelled.csv")
    a=ap.parse_args(); main(a.features,a.video,a.out)
