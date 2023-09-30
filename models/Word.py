class Word:

    def __init__(self, text, start, end, starting_delay, full_duration_until_next_word=None):
        self.text = text  # String contains the complete word in English characters
        self.start = start  # Integer. contains the millisecond value where word should be started to utter
        self.end = end  # Integer. contains the millisecond value where the uttered word should end

        self.starting_delay = starting_delay  # starting delay in milliseconds
        self.full_duration_until_next_word = full_duration_until_next_word  # duration remaining until next word starts