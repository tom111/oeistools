#!/usr/bin/env python3
from mido import Message, MidiFile, MidiTrack, MetaMessage
import numpy as np
import subprocess
from pydub import AudioSegment
import os
import tempfile
from pydub.silence import detect_nonsilent

# Define the sequence and its name as constants
SEQUENCE = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199, 211, 223, 227, 229, 233]
NAME = "primes"  # Name of the sequence (used for default filenames)

# SEQUENCE = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418]
# NAME = "fibonacci"

# SEQUENCE= [1, 1, 2, 5, 14, 42, 132, 429, 1430, 4862, 16796, 58786, 208012, 742900, 2674440, 9694845, 35357670, 129644790, 477638700, 1767263190, 6564120420, 24466267020]
# NAME = "catalan"

# SEQUENCE = [1, 2, 3, 4, 17, 6, 7, 8, 9, 17, 11, 12, 13, 14, 15, 16, 17, 18, 9, 20, 21, 22, 8, 24, 41, 12, 27, 28, 29, 30]
# NAME = "integers"

# SEQUENCE=[1, 1, 1, 1, 2, 3, 6, 11, 23, 47, 106, 235, 551, 1301, 3159, 7741, 19320, 48629, 123867, 317955, 823065, 2144505, 5623756, 14828074, 39299897, 104636890, 279793450, 751065460, 2023443032, 5469566585, 14830871802, 40330829030, 109972410221, 300628862480, 823779631721, 2262366343746, 6226306037178]
# NAME = "UnlabeledTrees" # https://oeis.org/A000055
# # Switches from low to high to fast because of the strong growth.

# SEQUENCE=[0, 4, 1, 10, 2, 16, 3, 22, 4, 28, 5, 34, 6, 40, 7, 46, 8, 52, 9, 58, 10, 64, 11, 70, 12, 76, 13, 82, 14, 88, 15, 94, 16, 100, 17, 106, 18, 112, 19, 118, 20, 124, 21, 130, 22, 136, 23, 142, 24, 148, 25, 154, 26, 160, 27, 166, 28, 172, 29, 178, 30, 184, 31, 190, 32, 196, 33]
# NAME="collatz" # https://oeis.org/A006370

# SEQUENCE=[1, 2, 1, 3, 10, 5, 16, 8, 4, 2, 1, 4, 2, 1, 5, 16, 8, 4, 2, 1, 6, 3, 10, 5, 16, 8, 4, 2, 1, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 8, 4, 2, 1, 9, 28, 14, 7, 22, 11, 34, 17, 52, 26, 13, 40, 20, 10, 5, 16, 8, 4, 2, 1, 10, 5, 16, 8, 4, 2, 1, 11, 34, 17, 52, 26, 13]
# NAME="collats_irreg_triangle" # https://oeis.org/A070165

# SEQUENCE=[0, 2, 1, 5, 2, 8, 3, 11, 4, 14, 5, 17, 6, 20, 7, 23, 8, 26, 9, 29, 10, 32, 11, 35, 12, 38, 13, 41, 14, 44, 15, 47, 16, 50, 17, 53, 18, 56, 19, 59, 20, 62, 21, 65, 22]
# NAME="collatzT" #https://oeis.org/A014682

# SEQUENCE=[1, 3, 1, 5, 9, 1, 9, 33, 31, 1, 17, 117, 221, 105, 1, 31, 429, 1465, 1473, 355, 1, 57, 1577, 10593, 18393, 9829, 1201, 1, 105, 5785, 76055, 260557, 230845, 65569, 4063, 1, 193]
# NAME="hilltopMaps" #https://oeis.org/A221446

# SEQUENCE=[-20, 56, 55, 56, 55, 56, 51, 54, 52, 49, 32, 37, 40, 44, 49, 51, 32, 36, 44, 48, 51, 52, 32, 37, 44, 56, 55, 56, 55, 56, 51, 54, 52, 49, 32, 37, 40, 44, 49, 51, 32, 36, 44, 52, 51, 49, 32, 37, -20, 56, 55, 56, 55, 56, 51, 54, 52, 49, 32, 37, 40, 44, 49, 51, 32, 36, 44, 48, 51, 52, 32, 37]
# NAME="fuerElise" #https://oeis.org/A123456

def compute_gaps(sequence):
    """Compute gaps between consecutive terms in the sequence."""
    return [sequence[i] - sequence[i-1] for i in range(1,len(sequence))]

def normalize_gaps(gaps, min_note=36, max_note=84):
    """
    Normalize gaps between sequence numbers to MIDI note range.
    
    Takes the gaps between consecutive numbers in a sequence and maps them to MIDI notes
    in a musically useful range. The smallest gap gets mapped to min_note (default 36/C2) 
    and the largest gap to max_note (default 84/C6), with all other gaps scaled linearly 
    in between.

    Args:
        gaps: List of integer gaps between consecutive sequence numbers
        min_note: Lowest MIDI note number to use (default 36/C2)
        max_note: Highest MIDI note number to use (default 84/C6)

    Returns:
        List of MIDI note numbers scaled between min_note and max_note.
        Returns empty list if gaps is empty.
        Returns list of middle C (60) if all gaps are equal.
    """
    if not gaps:
        return []
    
    min_gap = min(gaps)
    max_gap = max(gaps)
    
    # Handle the case where all gaps are the same
    if min_gap == max_gap:
        return [60] * len(gaps)  # Middle C for all notes - avoids division by zero
    
    # Linear mapping from gap range to note range
    # Formula: new_value = min_new + (value - min_old)/(max_old - min_old) * (max_new - min_new)
    normalized = []
    for gap in gaps:
        # Map gap to note range using linear interpolation
        print(f"gap: {gap}, min_gap: {min_gap}, max_gap: {max_gap}")
        note = min_note + ((gap - min_gap) / (max_gap - min_gap)) * (max_note - min_note)
        print(f"note: {note}")
        normalized.append(int(round(note)))  # Round to nearest MIDI note number
    
    return normalized

def create_synth_midi(gaps, filename=None):
    """Create a MIDI file with synthesizer sounds based on sequence gaps."""
    midi = MidiFile(ticks_per_beat=480)
    
    ## MELODY Track
    # Create a track for the melody (based on gaps)
    melody_track = MidiTrack()
    midi.tracks.append(melody_track)
    
    # Set tempo and instrument
    melody_track.append(MetaMessage('set_tempo', tempo=400000, time=0))  # 150 BPM
    melody_track.append(Message('program_change', program=80, time=0))  # Synth lead (program 80 = Lead 1 square)
    
    # Normalize gaps to MIDI note range
    notes = normalize_gaps(gaps)
    
    # Fixed note duration and spacing for consistent timing
    note_duration = 120  # Duration of each note
    note_spacing = 240   # Time between consecutive notes
    
    # Add notes based on gaps
    for i, note in enumerate(notes):
        # Vary velocity based on note value
        velocity = min(127, 70 + (note % 20))  # Higher notes slightly louder
        
        # Note on - starts playing the note with specified pitch, velocity (loudness), and timing
        # time=note_spacing ensures consistent rhythmic spacing between consecutive notes
        # This creates the actual sound event in the MIDI sequence
        melody_track.append(Message('note_on', note=note, velocity=velocity, time=note_spacing))
        
        # Add a harmony note to create a richer sound
        # For each melody note, we add either a perfect fifth (7 semitones) or perfect fourth (5 semitones)
        # The choice between fifth/fourth depends on the note value to add variety
        # Notes below middle C (value % 12 < 6) get a fifth, others get a fourth
        harmony_interval = 7 if note % 12 < 6 else 5  # Perfect fifth or perfect fourth
        harmony_note = note + harmony_interval
        
        # Only add harmony if it's within MIDI note range (0-127)
        if harmony_note < 127:
            # Add harmony note at slightly lower velocity (-20) for balance
            melody_track.append(Message('note_on', note=harmony_note, velocity=velocity-20, time=0))
            # Turn off harmony note after note_duration
            melody_track.append(Message('note_off', note=harmony_note, velocity=0, time=note_duration))
        
        # Turn off the main melody note
        melody_track.append(Message('note_off', note=note, velocity=0, time=0))

    # Add a final chord for resolution to the melody track
    final_note = notes[-1]
    final_chord = [final_note-12, final_note-5, final_note, final_note+7]

    # Extended hold time for the final chord
    extended_hold_time = note_spacing * 3
    
    # Add final chord with proper timing
    melody_track.append(Message('note_on', note=final_chord[0], velocity=90, time=note_spacing))
    for note in final_chord[1:]:
        if 0 <= note < 127:
            melody_track.append(Message('note_on', note=note, velocity=90, time=0))
    
    # Hold the final chord
    for i, note in enumerate(final_chord):
        if 0 <= note < 127:
            melody_track.append(Message('note_off', note=note, velocity=0, 
                                      time=extended_hold_time if i == 0 else 0))
    
    ## BASS Track
    # Create a track for the bass/rhythm
    bass_track = MidiTrack()
    midi.tracks.append(bass_track)
    bass_track.append(Message('program_change', program=38, time=0))  # Synth bass
    
    # Add bass notes with proper timing
    # The bass track will be shorter because it does not account for the final chord.
    bass_time = 0
    for i in range(0, len(notes), 4):  # One bass note every 4 melody notes
        if i < len(notes):
            bass_note = max(24, notes[i] - 24)  # One octave lower, but not too low
    
            # First bass note starts immediately, others after 4 note spacings
            bass_track.append(Message('note_on', note=bass_note, velocity=100, time= 0 if i==0 else note_spacing * 4))
    
            # Bass note duration
            bass_track.append(Message('note_off', note=bass_note, velocity=0, time=note_spacing * 2))

    
    ## PAD Track
    # Add a pad track for atmosphere
    pad_track = MidiTrack()
    midi.tracks.append(pad_track)
    pad_track.append(Message('program_change', program=90, time=0))  # Pad sound
    
    # Create chord progression based on the sequence
    unique_notes = sorted(set(notes))
    if len(unique_notes) >= 4:
        chord_notes = [unique_notes[0], unique_notes[len(unique_notes)//3], 
                      unique_notes[2*len(unique_notes)//3], unique_notes[-1]]
    else:
        chord_notes = [60, 67, 72, 76]  # Default chord if not enough unique notes
    
    # Calculate total piece duration
    total_melody_duration =  note_spacing * len(notes)
    
    # Create 4 chord sections
    section_count = min(4, len(notes) // 4)  # Ensure we have enough notes for sections
    section_length = total_melody_duration // section_count
    
    # Add chord sections with proper timing
    for section in range(section_count):
        # Select chord for this section
        root_note = chord_notes[section % len(chord_notes)]
        chord = [root_note, root_note + 4, root_note + 7]  # Simple triad
    
        # Start time for this section
        section_start_time = section * section_length if section > 0 else 0
    
        # Add chord notes
        for i, note in enumerate(chord):
            if 0 <= note < 127:
                pad_track.append(Message('note_on', note=note, 
                                       velocity=50, 
                                       time=section_start_time if i == 0 else 0))
    
        # Hold chord for section duration
        for i, note in enumerate(chord):
            if 0 <= note < 127:
                pad_track.append(Message('note_off', note=note, 
                                        velocity=0, 
                                        time=0))


    ## This outputs stats about the MIDI tracks        
    debug_midi_tracks(midi)
    
    if filename:
        midi.save(filename)
        print(f"MIDI file saved as {filename}")
    return midi

def debug_midi_tracks(midi):
    """Prints the events of each track in the MIDI file for debugging purposes and calculates track durations."""
    ticks_per_beat = midi.ticks_per_beat
    tempo = 400000  # microseconds per beat (150 BPM)
    seconds_per_tick = tempo / 1_000_000 / ticks_per_beat
    
    for i, track in enumerate(midi.tracks):
        print(f"Track {i}: {track.name if track.name else 'Unnamed'}")
        total_ticks = 0
        for msg in track:
            total_ticks += msg.time
            # if not msg.is_meta:
            #     print(f"{msg.type}: note={msg.note}, velocity={msg.velocity}, time={msg.time}")
            # else:
            #     print(f"Meta: {msg.type}, time={msg.time}")
        track_duration_seconds = total_ticks * seconds_per_tick
        print(f"Total duration for Track {i}: {track_duration_seconds:.2f} seconds")

def trim_silence(audio_segment, silence_thresh=-50.0, chunk_size=10):
    """Trim silence from the end of an audio segment."""
    non_silent_ranges = detect_nonsilent(audio_segment, min_silence_len=chunk_size, silence_thresh=silence_thresh)
    if non_silent_ranges:
        # Get the end of the last non-silent range
        end_trim = non_silent_ranges[-1][1]
        return audio_segment[:end_trim]
    return audio_segment

def sequence_to_mp3(sequence, name=NAME, mp3_filename=None, keep_intermediates=False, sf2_file="FluidR3_GM.sf2"):
    """Convert a sequence directly to MP3, optionally keeping intermediate files."""
    # Create output directory if it doesn't exist
    output_dir = 'music_outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Set default output filename based on sequence name if not provided
    if mp3_filename is None:
        mp3_filename = os.path.join(output_dir, f"{name.lower()}.mp3")
    
    # Create temporary files if not keeping intermediates
    if keep_intermediates:
        midi_file = os.path.join(output_dir, f"{name.lower()}.mid")
        wav_file = os.path.join(output_dir, f"{name.lower()}.wav")
    else:
        temp_dir = tempfile.gettempdir()
        midi_file = os.path.join(temp_dir, "temp_sequence.mid")
        wav_file = os.path.join(temp_dir, "temp_sequence.wav")
    
    # Compute gaps and create MIDI
    gaps = compute_gaps(sequence)
    
    # Handle very large gaps by capping them
    if gaps:
        max_reasonable_gap = 10000  # Set a reasonable maximum gap
        capped_gaps = [min(gap, max_reasonable_gap) for gap in gaps]
        create_synth_midi(capped_gaps, filename=midi_file)
    else:
        print("Warning: Sequence has no gaps (needs at least 2 elements)")
        return
    
    # Convert MIDI to WAV
    command = [
        "fluidsynth", 
        "-ni", sf2_file, midi_file, 
        "-F", wav_file, "-r", "44100"
    ]
    subprocess.run(command)
    # There is a general problem that the midi takes too long until it detects 
    # silence and ends the file.  Therefore we cut silence at the end of 
    # the while when makeing the mp3
    sound = AudioSegment.from_wav(wav_file)
    trimmed_sound = trim_silence(sound)
    trimmed_sound.export(mp3_filename, format="mp3", bitrate="192k")
    print(f"Created MP3 file: {mp3_filename}")
    
    # Clean up temporary files if not keeping intermediates
    if not keep_intermediates:
        try:
            os.remove(midi_file)
            os.remove(wav_file)
        except:
            pass  # Ignore errors in cleanup

# Main execution
if __name__ == "__main__":
    sequence_to_mp3(SEQUENCE, NAME, mp3_filename=None, keep_intermediates=True, sf2_file="FluidR3_GM.sf2")