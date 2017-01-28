import mido
mido.set_backend('mido.backends.rtmidi')
import time


class MidiControl(object):
    def __init__(self, midi_manager, name, control_number, value=0, min=0, max=255, allowed_values=None):
        self.midi_manager = midi_manager
        self.min = 0
        self.max = 255
        self.control_number = control_number
        self.value = value
        self.name = name
        self.allowed_values = allowed_values
        if self.allowed_values is not None:
            self.min = 0
            self.max = len(self.allowed_values) - 1
        midi_manager.register_control(self)

    def set_value(self, value):
        new_value = value / 127.0 * (self.max - self.min) + self.min

        if self.allowed_values is not None:
            self.value = self.allowed_values[int(new_value+0.01)]
        else:
            self.value = new_value

        print("Setting %s to %f" % (self.name, self.value))



class MidiManager(object):
    def __init__(self):
        self.inport = mido.open_input()
        self.controls = {}

    def register_control(self, control):
        self.controls[control.control_number] = control


    def poll(self):
        for msg in self.inport.iter_pending():
            if msg.type=='control_change':
                control_number = msg.control
                value = msg.value

                if control_number in self.controls:
                    self.controls[control_number].set_value(value)
                else:
                    print("Control_change: %d %d" % (control_number,value))
