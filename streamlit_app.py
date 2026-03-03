import streamlit as st
import pickle
import pandas as pd

# Load model
model = pickle.load(open("academic_warning_model.pkl", "rb"))

st.title( "HỆ THỐNG DỰ ĐOÁN CẢNH BÁO HỌC VỤ")

st.markdown("Nhập thông tin sinh viên để dự đoán trạng thái học tập.")

# ===== INPUT (HIỂN THỊ TIẾNG VIỆT) =====

tuoi = st.number_input("Tuổi", 17, 30, 20)

diem_danh_gia = st.number_input("Điểm đánh giá tổng hợp", 0.0, 100.0, 60.0)

so_mon_truot = st.number_input("Số môn bị điểm F", 0, 10, 0)

no_hoc_phi = st.selectbox("Tình trạng nợ học phí", [0, 1], 
                          format_func=lambda x: "Có nợ" if x == 1 else "Không nợ")

gioi_tinh = st.selectbox("Giới tính", ["Male", "Female"])

hinh_thuc_tuyen_sinh = st.selectbox("Hình thức tuyển sinh", 
                                    ["Exam", "Direct", "Transfer"])

tham_gia_clb = st.selectbox("Tham gia câu lạc bộ", [0, 1],
                            format_func=lambda x: "Có tham gia" if x == 1 else "Không tham gia")

nhan_xet_co_van = st.text_area("Nhận xét của cố vấn học tập")

bai_luan = st.text_area("Bài luận cá nhân")

# ===== PREDICT =====

if st.button("🔍 Dự đoán"):

    df_input = pd.DataFrame({
        "Age": [tuoi],
        "Training_Score_Mixed": [diem_danh_gia],
        "Count_F": [so_mon_truot],
        "Tuition_Debt": [no_hoc_phi],
        "Gender": [gioi_tinh],
        "Admission_Mode": [hinh_thuc_tuyen_sinh],
        "Club_Member": [tham_gia_clb],
        "Advisor_Notes": [nhan_xet_co_van],
        "Personal_Essay": [bai_luan],
        "combined_text": [nhan_xet_co_van + " " + bai_luan]
    })

    prediction = model.predict(df_input)[0]

    label_map = {
        0: "🟢 Bình thường",
        1: "🟡 Cảnh báo học vụ",
        2: "🔴 Nguy cơ thôi học"
    }

    st.success(f"Kết quả dự đoán: {label_map[prediction]}")
