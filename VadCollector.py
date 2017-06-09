import pdb
import os
import sys
import collections
import contextlib
import wave
import webrtcvad
from math import ceil, floor

class VadCollector(object):

    def __init__(self, audio_path=None, aggressiveness=0, framesize=30, padding_width=300, thresh=0.9):
        self.vad = webrtcvad.Vad(aggressiveness)
        self.framesize = framesize
        self.padding_width = padding_width
        self.thresh = thresh
        self.sample_width = 2
        if audio_path:
            self.set_audio(audio_path)
        num_padding_frames = int(self.padding_width / self.framesize)
        self.rt_buffer = collections.deque(maxlen=num_padding_frames)
        self.rt_triggered = False

    def rt_segmenter(self, cur_frame, audio_sample_rate):
        self.audio_sample_rate = audio_sample_rate
        is_voiced = self.vad.is_speech(cur_frame.bytes, self.audio_sample_rate)
        if not self.rt_triggered:
            self.rt_buffer.append([cur_frame, is_voiced])
            num_voiced = len([f[0] for f in self.rt_buffer if f[1]])
            if num_voiced > self.thresh * self.rt_buffer.maxlen:
                vad_start = self.rt_buffer[0][0].timestamp
                self.rt_triggered = True
                self.rt_buffer.clear()
                return (("vad_start", self.float_round(vad_start, 3)))
        else:
            self.rt_buffer.append([cur_frame, is_voiced])
            num_unvoiced = len([f[0] for f in self.rt_buffer if not f[1]])
            if num_unvoiced > self.thresh * self.rt_buffer.maxlen:
                # 70 sec trailing silence needed for phonetic identification
                vad_end = self.rt_buffer[int(70 / self.framesize)][0].timestamp
                self.rt_triggered = False
                self.rt_buffer.clear()
                return (("vad_end", self.float_round(vad_end, 3)))

    def clear_buffer(self):
        if self.rt_triggered:
            vad_end = self.rt_buffer[-1][0].timestamp
            self.rt_triggered = False
            self.rt_buffer.clear()
            return (("vad_end", self.float_round(vad_end, 3)))

    def set_audio(self, audio_path):
        au = AudioUtil()
        audio_data, self.audio_sample_rate = au.read_wave(
            audio_path, vad_assert=True)
        self.audio_frames = list(au.frame_generator(
            self.framesize, audio_data, self.audio_sample_rate, self.sample_width))

    def get_percentage(self):
        num_padding_frames = int(self.padding_width / self.framesize)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        frame_count = 0
        numerator = 0
        denominator = 0
        for frame in self.audio_frames:
            is_voiced = self.vad.is_speech(frame.bytes, self.audio_sample_rate)
            mess = ""
            if is_voiced:
                numerator += 1
            denominator += 1
            frame_count += 1
            if not triggered:
                ring_buffer.append([frame, is_voiced])
                num_voiced = len([f[0] for f in ring_buffer if f[1]])
                if num_voiced > self.thresh * ring_buffer.maxlen:
                    triggered = True
                    ring_buffer.clear()
            else:
                ring_buffer.append([frame, is_voiced])
                num_unvoiced = len([f[0] for f in ring_buffer if not f[1]])
                if num_unvoiced > self.thresh * ring_buffer.maxlen:
                    #vad_end = frame.timestamp + frame.duration
                    # need atleast 70msec for phonetic identification
                    triggered = False
                    ring_buffer.clear()
        if triggered:
            ring_buffer.clear()
        return float(numerator) / denominator

    def segment_bounds(self):
        vad_list = []
        num_padding_frames = int(self.padding_width / self.framesize)
        ring_buffer = collections.deque(maxlen=num_padding_frames)
        triggered = False
        frame_count = 0
        for frame in self.audio_frames:
            is_voiced = self.vad.is_speech(frame.bytes, self.audio_sample_rate)
            mess = ""
            if is_voiced:
                mess = "Frame " + str(frame_count) + " has  speech"
            else:
                mess = "Frame " + str(frame_count) + " doesn't have speech"
            print mess
            frame_count += 1
            if not triggered:
                ring_buffer.append([frame, is_voiced])
                num_voiced = len([f[0] for f in ring_buffer if f[1]])
                if num_voiced > self.thresh * ring_buffer.maxlen:
                    vad_start = ring_buffer[0][0].timestamp
                    triggered = True
                    ring_buffer.clear()
            else:
                ring_buffer.append([frame, is_voiced])
                num_unvoiced = len([f[0] for f in ring_buffer if not f[1]])
                if num_unvoiced > self.thresh * ring_buffer.maxlen:
                    #vad_end = frame.timestamp + frame.duration
                    # need atleast 70msec for phonetic identification
                    vad_end = ring_buffer[
                        int(70 / self.framesize)][0].timestamp
                    triggered = False
                    vad_list.append((self.float_round(vad_start, 3),
                                     self.float_round(vad_end, 3)))
                    ring_buffer.clear()
        if triggered:
            vad_end = frame.timestamp + frame.duration
            vad_list.append((self.float_round(vad_start, 3),
                             self.float_round(vad_end, 3)))
            ring_buffer.clear()
        return vad_list

    def excise(self, onset, offset):
        # onset and offset are in msec
        onset_frame = int(onset / self.framesize)
        offset_frame = int(offset / self.framesize)
        # print onset_frame, offset_frame
        excised_frames = self.audio_frames[onset_frame:offset_frame + 1]
        return b''.join([f.bytes for f in excised_frames])

    def extract_segments(self):
        vad_list = self.segment_bounds()
        for onset, offset in vad_list:
            yield (onset, offset, self.excise(onset * 1000, offset * 1000))

    def float_round(self, num, places=0, direction=floor):
        return direction(num * (10**places)) / float(10**places)


class Frame(object):

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


class AudioUtil(object):

    def frame_generator(self, frame_size_ms, audio, sample_rate, sample_width=1):
        n = int(sample_rate * (frame_size_ms / 1000.0) * float(sample_width))
        offset = 0
        timestamp = 0.0
        duration = (float(n) / sample_rate) / float(sample_width)
        while offset + n < len(audio):
            yield Frame(audio[offset:offset + n], timestamp, duration)
            timestamp += duration
            offset += n

    def read_wave(self, path, vad_assert=False):
        with contextlib.closing(wave.open(path, 'rb')) as wf:
            num_channels = wf.getnchannels()
            if vad_assert:
                assert num_channels == 1
            sample_width = wf.getsampwidth()
            if vad_assert:
                assert sample_width == 2
            sample_rate = wf.getframerate()
            if vad_assert:
                assert sample_rate in (8000, 16000, 32000)
            pcm_data = wf.readframes(wf.getnframes())
            return pcm_data, sample_rate

    def write_wave(self, path, audio, sample_rate, num_channels=1, sample_width=1):
        with contextlib.closing(wave.open(path, 'wb')) as wf:
            wf.setnchannels(num_channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio)


def main(args):
    if len(args) != 2:
        sys.stderr.write(
            'Usage: vadcollector.py <aggressiveness> <path to wav file>\n')
        sys.exit(1)
    framesize = 20  # in msec
    padding_width = 300  # in msec
    filename = os.path.basename(args[1])
    basename = os.path.splitext(filename)[0]
    vc = VadCollector(args[1], int(args[0]), framesize,
                       padding_width, thresh=0.9)
    segments = vc.extract_segments()
    for i, segment in enumerate(segments):
        onset = segment[0]
        offset = segment[1]
        segment = segment[2]
        path = basename + '_chunk-%002d.wav' % (i,)
        print(' Writing %s spanning %10.2f sec to %10.2f sec' %
              (path, onset, offset))
        AudioUtil().write_wave(path, segment, vc.audio_sample_rate,
                               num_channels=1, sample_width=2)


if __name__ == '__main__':
    main(sys.argv[1:])
