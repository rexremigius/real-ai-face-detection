import os
import numpy as np
import streamlit as st
import keras

os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"

st.set_page_config(page_title="Real or AI?", layout="centered")

N_ROUNDS = 10
NAME = {0: "AI", 1: "Real"}

@st.cache_resource
def load():
    model = keras.models.load_model("best_model.keras")
    data = np.load("test_images.npz")
    images, labels = data["images"], data["labels"].astype(int)
    scores = model.predict(images.astype("float32"), verbose=0).ravel()
    return images, labels, scores

images, labels, scores = load()
preds = (scores >= 0.5).astype(int)
s = st.session_state

def reset():
    s.order = np.random.choice(len(labels), N_ROUNDS, replace=False)
    s.round = 0
    s.human = 0
    s.model = 0
    s.guess = None
    s.history = []

def guess(g):
    i = s.order[s.round]
    s.guess = g
    s.human += int(g == labels[i])
    s.model += int(preds[i] == labels[i])
    s.history.append(
        {
            "Round": s.round + 1,
            "Answer": NAME[labels[i]],
            "You": NAME[g],
            "Model": NAME[preds[i]],
        }
    )

def next_image():
    s.round += 1
    s.guess = None

if "order" not in s:
    reset()

st.title("Real or AI?")
st.caption(
    "Guess whether each face is a real photo or AI-generated. The model guesses too."
)

played = len(s.history)
c1, c2, c3 = st.columns(3)
c1.metric("Round", f"{min(s.round + 1, N_ROUNDS)} / {N_ROUNDS}")
c2.metric("You", f"{s.human} / {played}")
c3.metric("Model", f"{s.model} / {played}")
st.progress(played / N_ROUNDS)

if s.round < N_ROUNDS:
    i = s.order[s.round]
    left, right = st.columns([3, 2], gap="large")

    with left:
        with st.container(border=True):
            st.image(images[i], use_container_width=True)

    with right:
        if s.guess is None:
            st.subheader("Your guess")
            st.button("Real", on_click=guess, args=(1,), use_container_width=True)
            st.button(
                "AI-Generated", on_click=guess, args=(0,), use_container_width=True
            )
        else:
            st.subheader(f"Answer: {NAME[labels[i]]}")

            if s.guess == labels[i]:
                st.success(f"You: {NAME[s.guess]} (correct)")
            else:
                st.error(f"You: {NAME[s.guess]} (wrong)")

            if preds[i] == labels[i]:
                st.success(f"Model: {NAME[preds[i]]} (correct)")
            else:
                st.error(f"Model: {NAME[preds[i]]} (wrong)")
            st.caption(f"Model score: {scores[i]:.2f}  (0 = AI, 1 = Real)")

            label = "See Results" if s.round + 1 == N_ROUNDS else "Next Image"
            st.button(
                label, on_click=next_image, type="primary", use_container_width=True
            )
else:
    st.divider()
    st.subheader("Final result")

    if s.human > s.model:
        st.success(f"You win, {s.human} to {s.model}.")
    elif s.model > s.human:
        st.error(f"The model wins, {s.model} to {s.human}.")
    else:
        st.info(f"It's a tie, {s.human} to {s.model}.")

    c1, c2 = st.columns(2)
    c1.metric("Your accuracy", f"{s.human / N_ROUNDS:.0%}")
    c2.metric("Model accuracy", f"{s.model / N_ROUNDS:.0%}")

    st.dataframe(s.history, hide_index=True, use_container_width=True)
    st.button("Play Again", on_click=reset, type="primary")
