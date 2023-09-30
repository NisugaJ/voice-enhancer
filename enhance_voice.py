# from functions.voice_enhancer import audio_plot
import sys

from functions.voice_enhancer.models.ProsodyWord import ProsodyWord
from functions.voice_enhancer.resemble_ai_connector.ssml_creator import calculate_prosodic_values
from functions.voice_enhancer.topic_extractor.LDA import run_lda_topic_extraction, load_data, \
    make_sentences_form_transcribed_data
from functions.voice_enhancer.resemble_ai_connector import ssml_creator
from functions.voice_enhancer.resemble_ai_connector import api
from utils.assembly_ai import speech_to_text
import time as time
# import audio_plot
from utils import constants
from components.subComponents import progressWindow
import random
from utils.converters.videoToAudioConverter import video_to_audio_converter
import tkinter.messagebox
import os
import uuid
from moviepy.editor import *

"""
    Lecturer Voice Enhancer
    @author : Nisuga Jayawardana (it18117110@my.sliit.lk)

    Objectives:
        1. Get the transcription of the video.
        2. Extract topic words from Topic Modelling - Gensim
        3. Set prosodic modifications to topic words using SSML (https://app.resemble.ai/docs#speech-synthesis-markup-language-ssml-reference)
        4. In the main transcription(SSML version), replace modified SSML elements with topic words
        5. Synthesise voice for the whole recording and generate an audio file from resemble.ai
        6. Replace the original audio of the video recording with the generated audio.

    Limitations:
        * The lecturer must have a cloned voice at resemble.ai. otherwise an available voice can be used.
        * The generated synthesized voice may not sync with the video
"""


class VoiceEnhancer:

    def __init__(self, video_path=None):
        self.video_path = None

        self.audio_file = None
        self.lecturer_transcription = None
        self.lecturer_transcription_file = None
        self.words_with_topic_biased_sorted = None

        self.sentences = []
        self.ssml_sentence_objects = []

        self.whole_transcription_in_ssml = None
        self.word_objects = []

        self.generated_audio_file = None
        # self.generated_word_timestamps = []

        self.topic_words_with_prosody = []

    def prepare_audio(self):
        if self.video_path is None or self.video_path == "":
            self.video_path = constants.new_video_import_path

        # Extracted audio of the video recording if not converted
        audio_file = constants.converted_audio_path
        if len(audio_file) <= 0:
            print("No previously extracted audio found.")
            print("Extracting the audio from the video....")
            conversion_ok = video_to_audio_converter(self.video_path, time.time())
            if not conversion_ok:
                message = "Failed to extract audio. Check whether a video is selected"
                print(message)
                tkinter.messagebox.showinfo(title='Error', message=message)
                return

        audio_file = constants.converted_audio_path
        self.audio_file = audio_file

    def __prepare_word_objects(self, word_objects):
        self.word_objects = word_objects

    def generate_transcription_file(self):
        if self.audio_file is None:
            tkinter.messagebox.showinfo(title='Error', message='Please convert to audio first !')
        else:
            # Generate lecturer's transcription of the audio and save to a file
            lecturer_transcription, word_objects = speech_to_text.get_transcription_from_assembly_ai(self.audio_file)
            # set word objects
            self.__prepare_word_objects(word_objects)
            transcription_file_name = str(uuid.uuid1()) + ".txt"
            current_dir = os.path.abspath(os.curdir)
            f_path = current_dir + "/AppData/tempStorage/voiceEnhancerTranscriptions/" + transcription_file_name
            print(f_path)
            f = open(f_path, "w")
            f.write(lecturer_transcription)
            f.close()
            print(f'transcription saved {f_path}')
            self.lecturer_transcription_file = f_path

    def prepare_transcription(self):
        if self.lecturer_transcription_file is None:
            tkinter.messagebox.showinfo(title='Error', message='Please generate the transcription first !')
        else:
            self.lecturer_transcription = load_data(self.lecturer_transcription_file)
            self.sentences = make_sentences_form_transcribed_data(self.lecturer_transcription)

    def run_topic_extraction(self, analyze=False):
        if self.lecturer_transcription is None:
            tkinter.messagebox.showinfo(title='Error', message='Please prepare the transcription first !')
        else:
            self.words_with_topic_biased_sorted = run_lda_topic_extraction(self.lecturer_transcription, analyze=analyze)

    def generate_ssml_content_per_sentence(self, gender=None):
        print(self.sentences)
        for index, sentence in enumerate(self.sentences):
            sentence_starting_index = 0
            sentence_length = len(sentence.split(' '))
            if index > 0:
                for pre_sentence in self.sentences[0: index]:
                    sentence_starting_index += len(pre_sentence.split(' '))

            word_objects_in_sentence = self.word_objects[
                                       sentence_starting_index:sentence_starting_index + sentence_length]
            if sentence != '' and sentence is not None:
                self.ssml_sentence_objects.append(
                    ssml_creator.get_ssml_for_sentence(sentence, self.words_with_topic_biased_sorted,
                                                       word_objects_in_sentence, gender=gender)
                )

    def generate_ssml_content_all(self):
        self.whole_transcription_in_ssml = ssml_creator.get_ssml_for_transcription(self.lecturer_transcription,
                                                                                   self.words_with_topic_biased_sorted,
                                                                                   self.word_objects)

    def generate_new_audio_using_sentences(self, speaker_uuid=None):
        print("starting")
        start_time = time.time()
        clip_creator = api.ClipCreator(self.sentences, self.ssml_sentence_objects, speaker_uuid=speaker_uuid)
        clip_creator.async_create_all_new_clips()
        duration = time.time() - start_time
        print(f"Downloaded {len(self.sentences)} in {duration} seconds")
        self.generated_audio_file = clip_creator.create_full_audio_output()
        return self.generated_audio_file

    '''
        Limited by resemble.api.
        Large SSML content is blocked.
    '''

    def generate_new_audio_using_whole_transcription(self):
        start_time = time.time()
        clip_creator = api.ClipCreator([self.lecturer_transcription], [self.whole_transcription_in_ssml])
        clip_creator.async_create_all_new_clips()
        duration = time.time() - start_time
        print(f"Downloaded single audio in {duration} seconds")
        self.generated_audio_file = clip_creator.create_full_audio_output()
        return self.generated_audio_file

    def replace_original_audio(self, save_to_dir=None):
        from datetime import datetime

        clip = VideoFileClip(self.video_path)
        generated_audi_clip = AudioFileClip(self.generated_audio_file)
        new_clip = clip.set_audio(generated_audi_clip)

        current_dir = os.path.abspath(os.curdir) if save_to_dir is None else save_to_dir
        now_time = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        new_file_name = now_time + "__export.mp4"
        f_path = current_dir + "/AppData/tempStorage/voiceEnhancer/newVideos/" + new_file_name
        new_clip.write_videofile(f_path, threads=8)

        # set new video to the main player
        constants.new_video_import_path = f_path

        return new_file_name

    def start_voice_enhancer(self):

        print("🎙🎙🎙 Starting Voice Enhancer 🎙🎙🎙")

        # Get extracted audio of the video recording
        self.prepare_audio()
        if self.audio_file is None:
            self.stop_voice_enhancer()
            return
        # audio_file = constants.converted_audio_path
        # if len(audio_file) <= 0:
        #     print("No extracted audio found.")
        #     print("Extracting the audio from the video....")
        #     conversion_ok = video_to_audio_converter(constants.new_video_import_path, time.time())
        #     if not conversion_ok:
        #         message = "Failed to extract audio. Check whether a video is selected"
        #         print(message)
        #         tkinter.messagebox.showinfo(title='Error', message=message)
        #         return
        #
        # audio_file = constants.converted_audio_path
        # print(audio_file)

        # Plot the audio - amplitude graph
        # audio_plot.get_audio_amp_graph()

        # Ask user to select a noisy part where no speaking happens.

        # Remove background noise using noisereduce py lib

        # Generate lecturer's transcription of the audio
        self.generate_transcription_file()
        # lecturer_transcription = speech_to_text.get_transcription_from_assembly_ai(audio_file)
        # transcription_file_name = str(uuid.uuid1())+".txt"
        # current_dir = os.path.abspath(os.curdir)
        # f_path = current_dir + "/AppData/tempStorage/voiceEnhancerTranscriptions/" + transcription_file_name
        # print(f_path)
        # f = open(f_path, "w")
        # f.write(lecturer_transcription)
        # f.close()
        # print(f'transcription saved {f_path}')

        # Load transcribed data
        self.prepare_transcription()
        # transcribed_text = load_data(f_path)

        # Get topic words with LDA output
        self.run_topic_extraction()
        # run_lda_topic_extraction(transcribed_text)

        # Setup up prosodic parameters for each topic word, based on it's topic value
        # Generate SSML elements for each topic word by applying the prosodic parameters
        # Format the transcription to a basic SSML document.
        # Replace the  topic words in the SSMl document, with the generated SSML elements
        # Validate the final SSML Document.
        self.generate_ssml_content_per_sentence()
        # self.generate_ssml_content_all()

        # send the final SSML Document resemble.ai API
        # Download the generated audio file
        new_audio_file_name = self.generate_new_audio_using_sentences()
        # new_audio_file_name = self.generate_new_audio_using_whole_transcription()

        # Ask user to preview the generated audio file.
        print(f'new_audio_file_name = {new_audio_file_name}')

        # If satisfied, Ask user to place the new audio file at a suitable starting point in the video timeline.
        # Else, cancel the process.

        # After user's confirmation, replace the audio with the new audio file.
        self.replace_original_audio()

    def calculate_topic_words_with_prosodic_attributes(self, gender='Male'):
        if len(self.words_with_topic_biased_sorted) == 0:
            tkinter.messagebox.showinfo(title='Error', message='Please run topic extraction before this!')
            return
        gender_symbol = 'M' if gender == 'Male' else 'F'
        print(f'Gender symbol is {gender_symbol}')

        for topic_obj_word in self.words_with_topic_biased_sorted.keys():
            print(f"topic: {topic_obj_word}")
            pitch, rate, volume = calculate_prosodic_values(self.words_with_topic_biased_sorted[topic_obj_word], gender=gender_symbol)
            self.topic_words_with_prosody.append(ProsodyWord(topic_obj_word, pitch, rate, volume))
        print(self.topic_words_with_prosody)

    def stop_voice_enhancer(self):
        self.video_path = None
        self.audio_file = None
        self.lecturer_transcription = None
        self.lecturer_transcription_file = None
        self.words_with_topic_biased_sorted = None
        self.sentences = None
        self.ssml_sentence_objects = []
        self.whole_transcription_in_ssml = None

        print('Voice enhancer stopped.')
