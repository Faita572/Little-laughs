import sys
import time

# Format: (Text_Chunk, delay_after_chunk_in_seconds, letter_speed)
script = [
    # Line 1: "Semua-mua yang ku mau"
    ("Semua-mua ", 0.1, 0.04),
    ("yang ", 0.08, 0.04),
    ("ku mau\n", 0.45, 0.04),

    # Line 2: "Ada padamu kok bisa gitu"
    ("Ada padamu, ", 0.15, 0.04),
    ("kok bisa ", 0.1, 0.04),
    ("gitu?\n", 0.5, 0.04),

    # Line 3: "A-aduh, pusing kepala, ci-cinta segitiga"
    ("A-aduh, ", 0.15, 0.035),
    ("pusing kepala, ", 0.2, 0.035),
    ("ci-cinta ", 0.1, 0.035),
    ("segitiga\n", 0.4, 0.035),

    # Line 4: "ku mau-mau aja jadi yang kedua"
    ("ku mau-mau aja ", 0.12, 0.035),
    ("jadi yang ", 0.1, 0.035),
    ("kedua\n", 0.55, 0.035),

    # Line 5 & 6: "Eh!!... ASTAGA! Bercanda!"
    ("Eh!!...", 0.35, 0.08),
    ("\nASTAGA! ", 0.2, 0.04),
    ("Bercanda!\n", 1.0, 0.04)
]

def print_lyrics(segments):
    for text, pause, char_speed in segments:
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(char_speed)
        time.sleep(pause)

if __name__ == "__main__":
    print_lyrics(script)