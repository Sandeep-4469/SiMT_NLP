from config import TranslationMode
from fixed_inference import translate as fixed_translate
from adaptive_inference import adaptive_inference
import sentencepiece as spm

sp = spm.SentencePieceProcessor(model_file="bpe.model")

mode_input = input("Choose mode (fixed/adaptive): ").strip().lower()

try:
    mode = TranslationMode(mode_input)
except ValueError:
    print("❌ Invalid mode. Please choose 'fixed' or 'adaptive'.")
    exit()

sentence = input("Enter German sentence: ").strip()

if mode == TranslationMode.FIXED:
    print("Translated (Fixed):", fixed_translate(sentence))
elif mode == TranslationMode.ADAPTIVE:
    print("Translated (Adaptive):", adaptive_inference(sp.encode(sentence)))
