"""
swim_feedback.py  -  Stage 3: new video features -> ranked timestamped feedback
Usage: python swim_feedback.py features/new_features.csv --profile cluster_out/well_profile.json
"""
import argparse, json, numpy as np, pandas as pd
TEXT={
 "trunk_angle_med":"Body angle {v:.0f} deg from flat (ideal ~{r:.0f}). Press chest down to raise hips.",
 "hip_depth_med":"Hips sitting low in the water. Lift the hips toward the surface.",
 "head_sub_frac":"Head below the surface {v:.0%} of the time. Keep the head back, airway clear.",
 "arm_speed":"Arm/hand speed differs from a steady float (yours {v:.2f} body-len/s).",
 "leg_speed":"Leg speed differs from a steady float (yours {v:.2f} body-len/s).",
 "leg_asym":"Left/right leg cadence uneven (asymmetry {v:.2f}). Balance the kick.",
 "arm_asym":"Left/right arm cadence uneven (asymmetry {v:.2f}). Balance the arms.",
 "trunk_angle_std":"Body angle unsteady (varies {v:.0f} deg). Hold a steadier position.",
}
def main(feat_csv, profile, out, z_thresh=1.5):
    prof=json.load(open(profile)); well=prof["well_profile"]; sd=prof["std"]
    df=pd.read_csv(feat_csv); rows=[]
    for _,w in df.iterrows():
        devs=[]
        for f in well:
            if f in w and not pd.isna(w[f]) and sd.get(f,0)>1e-6:
                z=(w[f]-well[f])/sd[f]
                if abs(z)>=z_thresh and f in TEXT: devs.append((abs(z),f,w[f],well[f]))
        devs.sort(reverse=True)
        for z,f,val,ref in devs[:3]:
            rows.append({"start_s":w.get("start_s"),"end_s":w.get("end_s"),"feature":f,
                         "z":round(z,1),"message":TEXT[f].format(v=val,r=ref)})
    fb=pd.DataFrame(rows); fb.to_csv(out,index=False)
    if len(fb):
        print("Most frequent issues across the video:")
        for f,n in fb["feature"].value_counts().head(5).items(): print(f"  {f}: flagged in {n} windows")
        print(f"\nfull timestamped feedback -> {out}\nexample windows:")
        for _,r in fb.head(6).iterrows(): print(f"  {r['start_s']:.0f}-{r['end_s']:.0f}s: {r['message']}")
    else: print("No windows deviated strongly from the 'well' profile.")
if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("features"); ap.add_argument("--profile",required=True)
    ap.add_argument("--out",default="feedback.csv"); ap.add_argument("--z",type=float,default=1.5)
    a=ap.parse_args(); main(a.features,a.profile,a.out,a.z)
