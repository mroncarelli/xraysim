from distutils.spawn import find_executable
import json
import os
import warnings

from .shared import instrumentsDir, defaultSixteCommand


def new_sw(message, category, filename, lineno, file=None, line=None):
    """
    Monkeypatch of warnings.showwarning
    :param message:
    :param category:
    :param filename:
    :param lineno:
    :param file:
    :param line:
    :return:
    """
    msg_list = '\n'.join(warnings.formatwarning(message, category, filename, lineno, line).split('\n')[0:-2]).split(':')
    print("WARNING:" + ':'.join(msg_list[3:]))


warnings.showwarning = new_sw


class Instrument:
    """
    This class define a Sixte instrument, with the attributes necessary to
    """

    def __init__(self, name: str, subdir: str, xml: str, command=None, special=None,
                 adv_xml=None, attitude=None):
        self.name = name
        self.subdir = subdir
        self.xml = xml
        self.command = command if command else defaultSixteCommand
        self.special = special
        self.adv_xml = adv_xml  # only Sixte version 2 or lower
        self.attitude = attitude

    def show(self):
        print("Name: " + self.name)
        print("Subdir: " + self.subdir)
        print("Xml: " + self.xml)
        if self.adv_xml:
            print("Adv. Xml: " + self.adv_xml)
        print("Command: " + self.command)
        if self.special:
            print("Special: " + self.special)
        if self.attitude:
            print("Attitude: " + self.attitude)

    def verify(self, warn=False, verbose=1) -> list:
        """
        Verifies if the instrument has been correctly initialized: checks if the subfolder exists and contains the xml
        files, checks if the command exists and if the special attribute corresponds to the implemented ones. Prints out
        a message containing "Instrument 'name' OK" or the list of wrong things.
        :param warn: (bool) If set the messages are issued as warnings.
        :param verbose (int) If > 0 messages are printed out, otherwise not
        :return: (list) Messages
        """
        messages = []
        path = instrumentsDir + '/' + self.subdir
        if not os.path.isdir(path):
            messages.append("Instrument " + self.name + " subdirectory does not exist: " + path)
        else:
            for xml in self.xml.split(','):
                full_xml = path + '/' + xml.strip()
                if not os.path.isfile(full_xml):
                    messages.append("Instrument " + self.name + " xml file does not exist: " + full_xml)
                if self.adv_xml is not None:
                    full_adv_xml = path + '/' + self.adv_xml
                    if not os.path.isfile(full_adv_xml):
                        messages.append("Instrument " + self.name + " xml file does not exist: " + full_xml)

        if find_executable(self.command) is None:
            messages.append("Instrument " + self.name + " command does not exist: " + self.command)

        if self.special:
            if self.special.strip().lower() not in [''] + special_list:
                messages.append("Instrument " + self.name + " special not recognized: " + self.special)

        ok = not messages
        if ok:
            messages = ["Instrument " + self.name + " OK"]

        if warn:
            if not ok:
                for msg in messages:
                    warnings.warn(msg)
        else:
            if verbose > 0:
                for msg in messages:
                    print(msg)

        return messages


def load_instrument(inp: dict) -> Instrument:
    """
    Initializes an instrument from a dictionary record.
    :param inp: (dict) Dictionary record usually derived from a JSON file
    """
    return Instrument(inp['name'], inp['subdir'], inp['xml'],
                      inp.get('command'), inp.get('special'), inp.get('adv_xml'), inp.get('attitude'))


class SixteInstruments:
    def __init__(self):
        self.__data = {}

    def keys(self):
        return self.__data.keys()

    def get(self, name) -> Instrument:
        return self.__data.get(name)

    def add(self, instrument: Instrument):
        self.__data[instrument.name] = instrument

    def load(self, inp):
        type_ = type(inp)
        if type_ == list:
            for d in inp:
                self.add(load_instrument(d))
        elif type_ == str:
            with open(inp) as file:
                json_data = json.load(file)
            self.load(json_data)
        else:
            raise ValueError("Invalid input type")

    def show(self, full=False):
        if self.__data == {}:
            print("No instrument loaded")
        else:
            print("List of available instruments:")
            for name in self.__data:
                if full:
                    instr = self.get(name)
                    print(" - " + name)
                    print("     Subdir: " + instr.subdir)
                    print("     Xml: " + instr.xml)
                    if instr.adv_xml:
                        print("     Adv. Xml: " + instr.adv_xml)
                    print("     Command: " + instr.command)
                    if instr.special:
                        print("     Special: " + instr.special)
                    if instr.attitude:
                        print("     Attitude: " + instr.attitude)

                else:
                    print("   - " + name)

    def verify(self, warn=False):
        if self.__data == {}:
            print("No instrument loaded")
        else:
            for name in self.__data:
                self.get(name).verify(warn=warn, verbose=1)
