import streamlit as st
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
import torch
import subprocess
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# Path to espeak-ng
espeak_path = r"C:\Program Files\eSpeak NG\espeak-ng.exe"

# Load the pretrained Wav2Vec2 model
bundle = WAV2VEC2_ASR_BASE_960H
model = bundle.get_model()
labels = bundle.get_labels()

# Title and description for the web app
st.title("Speech Phoneme Comparison Tool")
st.markdown("""
This tool allows you to compare phonemes from two audio files — a **reference** and a **test** — using the Wav2Vec2 ASR model. 
Upload your audio files and the system will transcribe, phonemize, and compare them.
""")

# File uploaders with clear instructions
st.subheader("Upload Audio Files")
st.markdown("""
- **Reference Audio**: The audio that contains the "correct" pronunciation.
- **Test Audio**: The audio you want to compare against the reference.
""")
reference_audio = st.file_uploader("Upload reference audio", type=["wav"])
test_audio = st.file_uploader("Upload test audio", type=["wav"])

def clean_transcription(transcription):
    """Clean the transcription by removing unwanted characters and ensuring proper spacing."""
    cleaned = re.sub(r'[^a-zA-Z\s]', '', transcription)  # Only keep alphabetic characters and spaces
    cleaned = ' '.join(cleaned.split())  # Remove extra spaces between words
    return cleaned

def save_uploaded_file(uploaded_file):
    """Save the uploaded file to disk."""
    # Create the temp_audio directory if it does not exist
    os.makedirs("temp_audio", exist_ok=True)

    # Save the file to the directory
    file_path = os.path.join("temp_audio", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def phonemize_with_espeak_ng(text):
    """Phonemize text using espeak-ng."""
    command = [espeak_path, "-v", "en", "-q", "-x", text]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip()

def process_audio(audio_file):
    """Convert audio to text and phonetic sequence."""
    # Save the uploaded file to disk
    file_path = save_uploaded_file(audio_file)

    # Load the audio file with the correct backend
    waveform, sample_rate = torchaudio.load(file_path, backend="soundfile")

    # Resample the audio to match the model's required sample rate
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)

    # Perform speech-to-text recognition
    with torch.inference_mode():
        emissions, _ = model(waveform)
        emissions = torch.log_softmax(emissions, dim=-1)  # Normalize probabilities
        token_ids = torch.argmax(emissions, dim=-1)[0].tolist()

    # Map IDs to characters using the model's labels
    transcription = "".join([labels[id] for id in token_ids]).replace("|", " ")  # Replace '|' with space

    # Clean the transcription to ensure proper format
    cleaned_transcription = clean_transcription(transcription)

    # Convert cleaned transcription to phonemes using espeak-ng
    phonemes = phonemize_with_espeak_ng(cleaned_transcription)  # Use subprocess to phonemize

    return cleaned_transcription, phonemes

def calculate_edit_distance(seq1, seq2):
    """Calculate edit distance and provide detailed feedback."""
    sm = SequenceMatcher(None, seq1, seq2)
    unmatched_parts = []

    for opcode in sm.get_opcodes():
        tag, i1, i2, j1, j2 = opcode
        if tag != 'equal':
            unmatched_parts.append((tag, seq1[i1:i2], seq2[j1:j2]))

    edit_distance = sum(1 for tag, *_ in unmatched_parts if tag != 'equal')

    return edit_distance, unmatched_parts

def generate_feedback(unmatched_parts):
    """Generate detailed feedback for unmatched phonemes."""
    feedback = []
    
    for tag, ref_part, test_part in unmatched_parts:
        if tag == "replace":
            feedback.append(f"Phoneme '{ref_part}' was replaced with '{test_part}'. Check articulation and transition.")
        elif tag == "delete":
            feedback.append(f"Phoneme '{ref_part}' is missing in the test utterance. Focus on articulating it clearly.")
        elif tag == "insert":
            feedback.append(f"Phoneme '{test_part}' was inserted in the test utterance. Check articulation and transition.")
    
    return feedback

def plot_spectrogram_with_annotations(audio_file, unmatched_parts, reference_phonemes, test_phonemes):
    """Plot the spectrogram and highlight unmatched phoneme parts."""
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file)
    if sample_rate != bundle.sample_rate:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=bundle.sample_rate)
        waveform = resampler(waveform)

    # Convert to mono by averaging channels if necessary
    if waveform.shape[0] > 1:  # Check if stereo
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # Average channels

    # Generate spectrogram
    spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=bundle.sample_rate)(waveform)

    # Apply log transformation and convert to NumPy array for plotting
    spectrogram_db = 10 * torch.log10(spectrogram + 1e-9).squeeze().numpy()

    # Plot the spectrogram
    plt.figure(figsize=(15, 5))
    plt.imshow(spectrogram_db, aspect='auto', origin='lower', cmap='viridis')
    plt.title("Spectrogram with Phoneme Annotations")
    plt.xlabel("Time")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Intensity (dB)")

    # Adjust number of time steps based on the phonemes
    num_reference_phonemes = len(reference_phonemes.split())
    num_test_phonemes = len(test_phonemes.split())

    # Set the number of time steps to the larger phoneme sequence
    num_phonemes = max(num_reference_phonemes, num_test_phonemes)

    time_steps = np.linspace(0, spectrogram_db.shape[1], num_phonemes + 1)  # Add 1 to ensure enough time steps

    for i, (tag, ref_part, test_part) in enumerate(unmatched_parts):
        if tag == "replace":
            color = "red"
        elif tag == "delete":
            color = "blue"
        elif tag == "insert":
            color = "orange"
        else:
            continue

        # Ensure annotation stays within bounds
        if i + 1 < len(time_steps):
            # Highlight the corresponding time step
            plt.axvspan(time_steps[i], time_steps[i + 1], color=color, alpha=0.3)
            plt.text(
                time_steps[i],
                spectrogram_db.shape[0] - 10,
                f"{tag}: {ref_part} → {test_part}",
                color=color,
                fontsize=9,
            )
        else:
            print(f"Warning: Skipping annotation for unmatched part at index {i} due to index out of bounds.")

    st.pyplot(plt)  # Display the plot in Streamlit

def display_comparison(reference_phonemes, test_phonemes, comparison_name="Comparison"):
    """Display the comparison results between reference and test phonemes."""
    # Calculate the edit distance and unmatched parts
    edit_distance, unmatched_parts = calculate_edit_distance(reference_phonemes, test_phonemes)

    # Display the comparison
    st.subheader(f"{comparison_name}: Reference vs Test")
    st.write(f"Edit Distance: {edit_distance}")
    
    # Display unmatched parts
    st.write("Unmatched Parts:")
    for tag, ref_part, test_part in unmatched_parts:
        st.write(f" - {tag.capitalize()} -> Reference: '{ref_part}', Test: '{test_part}'")
    
    # Generate feedback and display it
    feedback = generate_feedback(unmatched_parts)
    st.write("Feedback:")
    for f in feedback:
        st.write(f" - {f}")

# Ensure that both reference and test audio files are uploaded
# Ensure that both reference and test audio files are uploaded
if reference_audio is not None and test_audio is not None:
    # Process both reference and test audio files
    reference_transcription, reference_phonemes = process_audio(reference_audio)
    test_transcription, test_phonemes = process_audio(test_audio)

    # Display transcriptions and phonemes
    st.write("Reference Audio Transcription: ", reference_transcription)
    st.write("Reference Audio Phonemes: ", reference_phonemes)
    st.write("Test Audio Transcription: ", test_transcription)
    st.write("Test Audio Phonemes: ", test_phonemes)

    # Call the display function to show comparison results
    display_comparison(reference_phonemes, test_phonemes)

    # Calculate edit distance and unmatched parts for both phoneme sequences
    edit_distance, unmatched_parts = calculate_edit_distance(reference_phonemes, test_phonemes)

    # Plot the spectrogram with annotations
    plot_spectrogram_with_annotations(test_audio, unmatched_parts, reference_phonemes, test_phonemes)
else:
    st.write("Please upload both reference and test audio files.")

