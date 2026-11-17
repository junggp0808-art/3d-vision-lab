import streamlit as st
import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(page_title="3D Vision Lab", page_icon="🔭", layout="wide")
st.title("🔭 3D Vision Lab")
st.subheader("두 시점의 사진으로 3차원 특징점 복원하기")
st.write("보정된 스테레오 카메라로 촬영한 좌우 사진을 입력하면 대응 특징점을 찾아 3차원 좌표를 추정합니다.")

st.info("이 앱은 인체 표면 전체를 만드는 서비스가 아니라, 두 사진에서 대응되는 특징점의 희소 3차원 점군을 계산하는 교육용 실험 도구입니다.")

with st.sidebar:
    st.header("1. 입력 자료")
    calib_file = st.file_uploader("카메라 보정 파일 (.npz)", type=["npz"])
    left_file = st.file_uploader("왼쪽 카메라 사진", type=["jpg","jpeg","png"])
    right_file = st.file_uploader("오른쪽 카메라 사진", type=["jpg","jpeg","png"])
    max_features = st.slider("특징점 최대 개수", 500, 6000, 3000, 500)
    ratio = st.slider("특징점 매칭 엄격도", 0.5, 0.9, 0.7, 0.05)
    max_gap = st.number_input("두 시선 사이 허용 거리 (보정 단위)", min_value=0.01, max_value=10.0, value=1.0, step=0.05)
    run = st.button("3차원 복원 시작", type="primary", use_container_width=True)

st.markdown("""
### 2. 촬영 조건
- 좌우 사진은 **같은 장면을 서로 다른 위치에서** 촬영해야 합니다.
- 사람을 촬영한다면 촬영 중 자세와 표정이 크게 바뀌지 않도록 합니다.
- 카메라 보정에 사용한 카메라와 해상도·초점 설정을 유지합니다.
- 다른 사람의 사진은 반드시 동의를 받고 사용하세요.
""")

def read_image(upload):
    arr=np.frombuffer(upload.getvalue(), np.uint8)
    im=cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im is None:
        raise ValueError("이미지를 읽을 수 없습니다.")
    return im

def reconstruct(calib, left, right, max_features, ratio, max_gap):
    required=["K1","D1","K2","D2","R","T"]
    if not all(k in calib for k in required):
        raise ValueError("보정 파일에 K1, D1, K2, D2, R, T 항목이 필요합니다.")
    K1,D1,K2,D2,R,T=[calib[k] for k in required]
    T=np.asarray(T).reshape(3)
    if left.shape[:2] != right.shape[:2]:
        raise ValueError("좌우 사진의 크기가 서로 다릅니다.")
    gray1=cv2.cvtColor(left,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(right,cv2.COLOR_BGR2GRAY)
    orb=cv2.ORB_create(nfeatures=int(max_features))
    kp1,des1=orb.detectAndCompute(gray1,None)
    kp2,des2=orb.detectAndCompute(gray2,None)
    if des1 is None or des2 is None:
        raise ValueError("특징점을 찾지 못했습니다. 질감이나 모서리가 있는 장면을 촬영하세요.")
    matcher=cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs=matcher.knnMatch(des1,des2,k=2)
    good=[m for pair in pairs if len(pair)==2 for m,n in [pair] if m.distance < ratio*n.distance]
    if len(good)<8:
        raise ValueError(f"유효한 특징점 매칭이 부족합니다: {len(good)}개")
    p1=np.float32([kp1[m.queryIdx].pt for m in good])
    p2=np.float32([kp2[m.trainIdx].pt for m in good])
    n1=cv2.undistortPoints(p1.reshape(-1,1,2),K1,D1).reshape(-1,2)
    n2=cv2.undistortPoints(p2.reshape(-1,1,2),K2,D2).reshape(-1,2)
    F,mask=cv2.findFundamentalMat(n1,n2,cv2.FM_RANSAC,0.001,0.99)
    if F is None or mask is None:
        raise ValueError("대응점 기하 검증에 실패했습니다.")
    keep=mask.ravel().astype(bool)
    n1,n2=n1[keep],n2[keep]
    C1=np.zeros(3)
    C2=-R.T@T
    pts=[]; gaps=[]; angles=[]
    for a,b in zip(n1,n2):
        ray1=np.array([a[0],a[1],1.0])
        ray2=R.T@np.array([b[0],b[1],1.0])
        u=ray1/np.linalg.norm(ray1)
        v=ray2/np.linalg.norm(ray2)
        w=C1-C2
        c=float(np.dot(u,v)); d=float(np.dot(w,u)); e=float(np.dot(w,v))
        den=1-c*c
        if den<1e-8: continue
        s=(c*e-d)/den
        t=(e-c*d)/den
        if s<=0 or t<=0: continue
        P1=C1+s*u; P2=C2+t*v
        gap=float(np.linalg.norm(P1-P2))
        if gap>max_gap: continue
        pts.append((P1+P2)/2)
        gaps.append(gap)
        angles.append(float(np.degrees(np.arccos(np.clip(abs(c),0,1)))))
    if not pts:
        raise ValueError("복원된 점이 없습니다. 사진 쌍과 카메라 보정 파일을 확인하세요.")
    df=pd.DataFrame(np.asarray(pts),columns=["X","Y","Z"])
    df["ray_gap"]=gaps
    df["ray_angle_deg"]=angles
    return df,len(good),int(keep.sum())

if run:
    if not (calib_file and left_file and right_file):
        st.error("보정 파일과 좌우 사진을 모두 업로드해 주세요.")
    else:
        try:
            calib=np.load(calib_file)
            left=read_image(left_file); right=read_image(right_file)
            with st.spinner("특징점 분석 및 3차원 좌표 계산 중..."):
                df,nmatch,ninlier=reconstruct(calib,left,right,max_features,ratio,max_gap)
            st.success("3차원 특징점 복원이 완료되었습니다.")
            c1,c2,c3=st.columns(3)
            c1.metric("초기 매칭점",nmatch)
            c2.metric("기하 검증 통과",ninlier)
            c3.metric("최종 복원 점",len(df))
            fig=go.Figure(data=[go.Scatter3d(
                x=df.X,y=df.Y,z=df.Z,mode="markers",
                marker=dict(size=3,color=df.Z,colorscale="Viridis",opacity=0.85),
                text=[f"X={r.X:.2f}, Y={r.Y:.2f}, Z={r.Z:.2f}" for r in df.itertuples()],
                hoverinfo="text")])
            fig.update_layout(title="복원된 3차원 특징점",scene=dict(
                xaxis_title="X (보정 단위)",yaxis_title="Y (보정 단위)",zaxis_title="Z (보정 단위)"),
                height=650,margin=dict(l=0,r=0,b=0,t=45))
            st.plotly_chart(fig,use_container_width=True)
            st.subheader("복원 좌표")
            st.dataframe(df, use_container_width=True)
            st.download_button("CSV 결과 다운로드",df.to_csv(index=False).encode("utf-8-sig"),
                               file_name="3d_coordinates.csv",mime="text/csv")
            st.caption("좌표 단위는 카메라 보정 때 체커보드 한 칸에 지정한 단위(cm 등)를 따릅니다.")
        except Exception as e:
            st.error(str(e))

st.divider()
st.caption("교육·연구용 데모 | 사진의 품질, 카메라 보정, 특징점 매칭에 따라 결과가 달라집니다.")
