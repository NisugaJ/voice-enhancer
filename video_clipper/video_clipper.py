import os
import uuid

import moviepy.editor as moviepy_editor

from functions.voice_enhancer.models.Word import Word
from utils.assembly_ai import speech_to_text

current_dir = os.path.abspath(os.curdir)
video_clipper_dir = current_dir + "/AppData/tempStorage/voiceEnhancer/videoClipper/"


class VideoClipper:

    def __init__(self, video_path=None, save_dir=None):
        self.video_path = video_path
        if self.video_path is not None:
            self.video_clip = moviepy_editor.VideoFileClip(self.video_path)
        else:
            self.video_clip = None

        # create a mew directory
        self.clipper_uuid = str(uuid.uuid1())
        self.save_dir = (save_dir if save_dir is not None else video_clipper_dir) + self.clipper_uuid
        try:
            os.mkdir(self.save_dir)
        except:
            OSError("Could not create folder for video clipper: {}".format(self.clipper_uuid))

        self.word_objects = []
        self.synthesized_word_objects = None
        self.clips_data = [
            # {
            #     "original": {
            #         "word": Word,
            #         "end_frame": str
            #     },
            #     "synthesized": {
            #         "word": Word,
            #         "start_frame": str
            #         "end_frame": str
            #     }
            # }
        ]

    def create_frames(self):
        if self.video_clip is not None:
            # getting only first 5 seconds
            clip = self.video_clip.subclip(0, 5)
            clip.write_images_sequence(self.save_dir + "frame%04d.jpeg", fps=24)
            # with open(self.save_dir+"2", 'w') as f:
            #     f.write(frame)

    def set_words(self, words):
        self.word_objects = words

    def detect_frozen_zones(self):
        import cv2
        import numpy as np
        cv2_clip = cv2.VideoCapture(self.video_path)
        second_counts = 0
        while True:
            time1 = second_counts  # This is current second
            time2 = second_counts + 1000  # This is the next time in second
            second_counts = time2  # increment to next time

            cv2_clip.set(cv2.CAP_PROP_POS_MSEC, time1)
            try:
                ret, Frame1 = cv2_clip.read()
                if Frame1 is None:
                    break

                cv2_clip.set(cv2.CAP_PROP_POS_MSEC, time2)
                ret, Frame2 = cv2_clip.read()
                if Frame2 is None:
                    break

                Diff = np.sum(np.abs(Frame1 - Frame2))
                print(f'{Diff}, {time1 / 1000}, {time2 / 1000}')

            except Exception as e:
                print(e)
                break

    def create_word_clips(self):
        if self.word_objects is not None:
            for index, word_object in enumerate(self.word_objects):
                start = word_object.start / 1000
                end = (
                        word_object.full_duration_until_next_word / 1000 + start) if word_object.full_duration_until_next_word is not None \
                    else word_object.end / 1000
                print(f'start: %s end: %s' % (start, end))
                clip = self.video_clip.subclip(start, end)
                clip.write_images_sequence(self.save_dir + '/{}_%04d.png'.format(index), fps=24)

    def generate_word_objects_from_synthesized(self, generated_word_timestamps):

        # create word array from generated_word_timestamps
        synzed_words = []
        for sentence_index, sentence in enumerate(generated_word_timestamps):
            for key, value in sentence.items():
                synzed_words += value

        print(synzed_words)
        for word in self.word_objects:
            print(f'w-> {word.text}, s={word.start}, e={word.end}')

        self.synthesized_word_objects = synzed_words
        print(f"length of synzed_words = {len(synzed_words)}")
        print(f"length of word_objects = {len(self.word_objects)}")

    def prepare_clip_data(self):
        matched = 0
        unmatched = 0
        for index, original_word in enumerate(self.word_objects):
            if index < len(self.synthesized_word_objects):
                synthesized_word = self.synthesized_word_objects[index]
                if synthesized_word.text.lower() == original_word.text.lower():
                    print(f'{index} matching. synzed={synthesized_word.text}, ori={original_word.text}')
                    clip_data = {
                        "original": {
                            "word": original_word,
                            "start_frame": 'str',
                            "end_frame": 'str',
                        },
                        "synthesized": {
                            "word": synthesized_word,
                            "start_frame": 'str',
                            "end_frame": 'str'
                        }
                    }
                    matched += 1
                else:
                    print(f'{index} not matching. synzed={synthesized_word.text}, ori={original_word.text}')
                    self.synthesized_word_objects \
                        = self.synthesized_word_objects[:index] \
                          + [None] \
                          + self.synthesized_word_objects[index + 1:]
                    print(f'len {len(self.synthesized_word_objects)}')
                    unmatched += 1

        print(f'matched={matched}, unmatched={unmatched}')

    # def search_top_word(self, word):
