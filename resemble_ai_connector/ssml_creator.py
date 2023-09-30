from ssml_builder.core import Speech


def create_speaker_element(inner_content):
    speech = Speech()
    speech.add_text(inner_content)
    print(speech.speak())
    return speech


"""
Creates a prosody tag

Prosody tag
An optional tag used to style the way synthesized speech sounds by specifying the pitch, rate, or volume.

Syntax

<prosody pitch="string" rate="string" volume="string"></prosody>

pitch: The baseline pitch of the synthesized speech. This must be one of the following values:
        x-Low
        low
        medium
        high
        x-high

rate: The baseline speed of the synthesized speech. The rate must be a percent value. For example 100% is normal, 50% is half as fast as normal, 
      200% is double the speed of normal.

volume: Indicates the volume level of the synthesized speech. This must be one of the following values:

        silent
        x-soft
        soft
        medium
        loud
        x-loud

Ex: <prosody rate="150%" pitch="high" volume="loud">Text content</prosody>
"""
def create_prosody_element(content, probability, gender='M'):
    speech = Speech()
    pitch, rate, volume = calculate_prosodic_values(probability, gender)
    speech.prosody(value=content, pitch=pitch, volume=volume, rate=rate)

    return speech.speak().replace("<speak>", '').replace("</speak>", '')


def get_ssml_for_transcription(transcription, words_with_topic_biased, word_objects):
    content = ""

    transcription_splitted = transcription.split(' ')
    print(transcription_splitted)
    for index, word in enumerate(transcription_splitted):

        #  add breaks between words
        word_delay = int(round( word_objects[index].starting_delay / 1000, 0))
        if word_delay > 0:
            content += '<break time="{}s"/>'.format(word_delay)
            print(content)

        if word in words_with_topic_biased.keys():
            probability = words_with_topic_biased[word]
            print(f'{word}: {probability}')
            content += create_prosody_element(
                content=word,
                probability=probability,
            )
        else:
            content += " " + word + " "
    return create_speaker_element(inner_content=content)


def get_ssml_for_sentence(sentence, words_with_topic_biased, word_objects, gender):
    content = ""
    #
    # last_char = sentence[-1]
    # if last_char == '.':
    #     content = '<resemble:emotion emotions="happy" >{}</emotion>'.format(sentence)
    # elif last_char == '?':
    #     content = '<resemble:emotion emotions="question" >{}</emotion>'.format(sentence)

    transcription_splitted = sentence.split(' ')

    for index, word in enumerate(transcription_splitted):

        #  add delays between words
        word_delay = word_objects[index].starting_delay / 1000  # converting to seconds
        if word_delay > 0:
            # content += '<break time="{}s"/>'.format(word_delay)
            # content += '<prosody volume="silent" duration="{}s">test</prosody>'.format(5)
            # content += '<prosody volume="silent">test</prosody>'
            if word_delay > 0.690:
                content += ',,,,'
            elif word_delay > 0.480:
                content += ',,,'
            elif word_delay > 0.140:
                content += ',,'
            elif word_delay > 0.040:
                content += ','
            print(content)

        prosody_content = ""
        if word in words_with_topic_biased.keys():
            probability = words_with_topic_biased[word]
            print(f'{word}: {probability}')
            prosody_content += create_prosody_element(
                content=word,
                probability=probability,
                gender=gender
            )
            content += prosody_content
        else:
            content += " " + word + " "
    return create_speaker_element(inner_content=content)


'''
Calculate prosodic values using below explanation

p = Probability of relativity of a word to default topic of the corpus

Personality dimension	Pitch level (pitch)	  Tempo (rate)	    Loudness (volume)
Excited (female)	    +1 * p	              +1 * p	        +1 * p
Competence (male)	    -1 * p	              +1 * p	        +1 * p


    Parameters
    ----------
    probability :  Average Probability of relativity of a word to all topics modelled from the transcription.
                   Between 0 and 1

    gender : F for `Female`. M for `Male`

    Returns
    -------
    pitch, rate, volume

'''


def calculate_prosodic_values(probability, gender):
    global pitch, rate, volume

    print(f'Calculating prosodic values for probability: {probability}, gender: {gender}')
    rate = str(
        int(round(100 + (probability * 100), 0))
    ) + "%"
    print(f'calculated rate: {rate}')

    if probability > 0.1:
        volume = "loud"
    else:
        volume = "medium"
    print(f'calculated volume: {volume}')

    if gender == 'F':
        pitch = "high"

    elif gender == 'M':
        pitch = "low"

    else:
        print(f'Invalid Gender detected. Use F for Female and M for Male.')
    print(f'calculated pitch: {pitch}')

    return pitch, rate, volume

# prosody_speech = create_prosody_element()
# create_speaker_element(prosody_speech)
