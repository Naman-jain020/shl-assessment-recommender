import streamlit as st, requests, os

API = os.getenv("API_URL", "http://localhost:8000")

st.title("SHL Assessment Recommender")
txt = st.text_area("Paste a JD or type a natural-language query:", height=200,
    placeholder="Need a Java developer who is good in collaborating with external teams and stakeholders.")
topk = st.slider("How many results?", 1, 10, 10)

col1, col2 = st.columns(2)
if col1.button("Health"):
    r = requests.get(f"{API}/health").json()
    st.write(r)

if col2.button("Recommend") and txt.strip():
    r = requests.post(f"{API}/recommend", json={"text": txt, "k": topk})
    if r.status_code == 200:
        out = r.json()["results"]
        for i,rec in enumerate(out,1):
            st.markdown(f"**{i}. [{rec['name']}]({rec['url']})**  \nType: `{rec['test_type']}` · Score: `{rec['score']:.4f}`")
    else:
        st.error(f"API error: {r.status_code} - {r.text}")