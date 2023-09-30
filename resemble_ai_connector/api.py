import json

import numpy
import requests
import time
import concurrent.futures
import threading
import multiprocessing
import os
import uuid
from pydub import AudioSegment

from decorator import __init__

api_base_url = "https://app.resemble.ai/api/v1"
callback_service_base_url = "http://2021-064-voice-enhancer-helper.azurewebsites.net"
current_dir = os.path.abspath(os.curdir)
folder_path = current_dir + "/AppData/tempStorage/voiceEnhancer/clips/"
project_uuid = "bc5187f6"
voice_nisuga = "6b5ad3e7"
voice_jordan = "9c930ee9"
voice_tanja = "16c50f71"
voice_excited_tanja = "584248ee"

headers = {
    'Authorization': 'Token token=LP4y5RanbIQHxan3k5zLRQtt',
    'Content-Type': 'application/json'
}

thread_local = threading.local()


def get_session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def get_projects():
    url = api_base_url + "/projects"
    response = requests.get(url, headers=headers)
    print(response.json())
    return response.json()


def get_a_specific_project(project_uuid):
    url = api_base_url + "/projects/" + str(project_uuid)
    response = requests.get(url, headers=headers)
    print(response.json())
    return response.json()


def get_all_voices():
    url = api_base_url + "/voices"
    response = requests.get(url, headers=headers)
    print(response.json())
    return response.json()


def sync_create_a_new_clip(project_uuid, ssml):
    url = api_base_url + "/projects/" + project_uuid + "/clips/sync"
    data = {
        'data': {
            'title': 'Episode 1',
            'body': ssml,
            'voice': '6b5ad3e7',
        }
    }
    response = requests.post(url, headers=headers, json=data)
    print(response.json())
    return response.json()


class ClipCreator:

    def __init__(self, sentences, ssml_speak_objects, speaker_uuid=None):
        self.sentences = sentences
        self.ssml_speak_objects = ssml_speak_objects
        self.clips = [None] * len(self.sentences)  # file names of generated audio clips. Initialized by None array
        # of length of sentences

        self.transcription_id = str(uuid.uuid1())
        self.file_path = ""
        self.create_folder()

        self.speaker_uuid = speaker_uuid if speaker_uuid is not None else voice_jordan

    def create_folder(self):
        file_path = folder_path + self.transcription_id + "/"
        try:
            os.mkdir(file_path)
            self.file_path = file_path
        except OSError as error:
            print(error)

    def async_create_a_new_clip(self, index, item):
        clip_title = item["sentence"]
        ssml = item["ssml"]

        session = get_session()
        url = api_base_url + "/projects/" + project_uuid + "/clips"
        data = {
            'data': {
                'title': clip_title,
                'body': ssml,
                'voice': self.speaker_uuid,
            },
            'callback_uri': callback_service_base_url + "/clips/accept-callback"
        }
        response = session.post(url, headers=headers, json=data)
        print(response.json())
        clip_id = response.json()["id"]
        print("clip_id", clip_id)

        def get_clip_data():
            url = callback_service_base_url + "/clips/" + clip_id
            print(f'url: {url}')
            clip_response = session.get(url)
            if clip_response.status_code == 200:
                return clip_response.json()
            else:
                return {"error": True, "message": "Clip is not ready"}

        clip_response = {"is_clip_generated": False}
        if response.status_code == 200:
            is_clip_generated = False
            while is_clip_generated is not True:
                print(ssml)
                clip_response = get_clip_data()
                if clip_response.get("url") is not None:
                    clip_url = clip_response.get("url")
                    print("audio_url:" + clip_url)

                    # download the clip
                    data = session.get(clip_url)
                    # save the clip to local
                    file_path = self.file_path + index + ".wav"
                    with open(file_path, 'wb') as file:
                        file.write(data.content)
                        print(f"Saved clip. {index}")
                        self.clips[int(index)] = index + ".wav"

                    is_clip_generated = True

                    # add new items to the clip_response
                    clip_response['sentence_index'] = int(index)
                    end_times = (clip_response.get("phoneme_timestamps"))["end_times"]
                    phoneme_chars = (clip_response.get("phoneme_timestamps"))["phoneme_chars"]
                    words_with_times = self.calculate_chars_with_end_times_for_word(end_times, phoneme_chars)
                    clip_response['words_with_times'] = words_with_times
                else:
                    print("audio_url still not created")
                    time.sleep(3)

        return clip_response

    def async_create_all_new_clips(self):
        iter = {i: {"sentence": self.sentences[i], "ssml": self.ssml_speak_objects[i].speak()} for i in
                range(len(self.sentences))}
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            results = executor.map(self.async_create_a_new_clip, [str(i) for i in iter.keys()], iter.values())
            # all_sentences_with_word_times = []
            # for result in results:
            #     all_sentences_with_word_times.append({
            #         result['sentence_index']: result['words_with_times']
            #     })
            # print(all_sentences_with_word_times)
            # return self.calculate_new_word_timestamps(all_sentences_with_word_times)


    def create_full_audio_output(self):
        if len(self.clips) == 1:
            print("Only one clip detected.")
            return self.clips[0]

        # import pydub

        audio_segments = []

        for clip in self.clips:
            # collect all audio clips as AudioSegments
            audio_segments.append(
                AudioSegment.from_file(self.file_path + clip, format="wav")
            )

        # Summation of AudioSegments
        combined = sum(audio_segments)

        # Generating final audio of summation
        file_path = self.file_path + self.transcription_id + "_FINAL.wav"
        file_handle = combined.export(file_path, format="wav")
        print(f"Saved final audio: {file_path}")

        return file_path

    def calculate_chars_with_end_times_for_word(self, end_times, phoneme_chars):

        chars_with_end_times = {end_times[i]: phoneme_chars[i] for i in
                                range(len(end_times))}

        print(chars_with_end_times)
        words = []
        word_start_time = None
        word_end_time = None
        index = 0
        char_keys = list(chars_with_end_times)
        for index, (end_time, phoneme_char) in enumerate(chars_with_end_times.items()):
            # print(f'{end_time}: {chars_with_end_times[end_time]}')
            current__phoneme_char = chars_with_end_times[end_time]
            if current__phoneme_char == '<w>':
                word_start_time = end_time
            if current__phoneme_char == '</w>':
                word_end_time = end_time
                if index + 1 == len(chars_with_end_times) - 1:
                    word_end_time = char_keys[index+1]
                    print(f"at end of sentence {word_end_time}: {chars_with_end_times[char_keys[index+1]]}")
            if word_start_time is not None and word_end_time is not None:
                words.append([word_start_time, word_end_time])
                word_start_time = None
                word_end_time = None

        return words

    def calculate_new_word_timestamps(self, all_sentences_with_word_times):
        # print(f'Before sorting: {all_sentences_with_word_times}')
        all_sentences_with_word_times = sorted(all_sentences_with_word_times, key=lambda d: list(d.keys()))
        # print(f'After sorting: {all_sentences_with_word_times}')
        prev_sentence_ending_time = 0.0
        for sentence_index, sentence in enumerate(all_sentences_with_word_times):
            for key, value in sentence.items():
                current_sentence_ending_time = float(value[-1][-1])

                new_sentence = []
                for index, word in enumerate(value):
                    word[0] = str(float(word[0]) + float(prev_sentence_ending_time))
                    word[1] = str(float(word[1]) + float(prev_sentence_ending_time))
                    new_sentence.append(word)
                all_sentences_with_word_times[sentence_index][key] = new_sentence

                # increase the sentence ending time
                print(f'current_sentence_ending_time: {current_sentence_ending_time}')
                prev_sentence_ending_time += current_sentence_ending_time
                print(f'prev_sentence_ending_time: {prev_sentence_ending_time}')
        print(f'ALL: \n {all_sentences_with_word_times}')

        return all_sentences_with_word_times
