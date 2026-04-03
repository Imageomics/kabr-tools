"""
Package to manage focal data from Animal Behavior Pro
"""
import csv
import collections
import dataclasses
import datetime


def _read_csv(file) -> "dict[str, str]":
    with open(file) as fd:
        reader = csv.DictReader(fd)
        return list(reader)


def _clean_value(raw_value):
    return raw_value.strip('" ')


def _parse_absolute_time(ymd, hms):
    ymd = _clean_value(ymd)
    hms = _clean_value(hms)
    return datetime.datetime.strptime(ymd + "|" + hms, "%Y-%m-%d|%H:%M:%S")


@dataclasses.dataclass(frozen=True)
class Action:
    name: str
    absolute_start: datetime.datetime
    relative_start: datetime.timedelta
    duration: datetime.timedelta

    @classmethod
    def from_row(cls, row, conversion=None):
        name = _clean_value(row[" Behavior"])
        if callable(conversion):
            name = conversion(name)

        absolute_start = _parse_absolute_time(
            row["Date_ymd"], row[" Time_Absolute_hms"]
        )

        relative_start = datetime.timedelta(seconds=float(row[" Time_Relative_s"]))
        duration = datetime.timedelta(seconds=float(row[" Duration_s"]))

        return cls(name, absolute_start, relative_start, duration)

    @property
    def absolute_stop(self):
        return self.absolute_start + self.duration

    @property
    def relative_stop(self):
        return self.relative_start + self.duration

    def relative_slice(self, start_time, stop_time):
        """
        Returns a copy of this action starting from start_time and ending at stop_time
        """
        absolute_start = self.absolute_start

        relative_start = self.relative_start
        if start_time > relative_start:
            absolute_start += start_time - relative_start
            relative_start = start_time

        stop_time = min(stop_time, self.relative_stop)
        duration = stop_time - relative_start

        return self.__class__(self.name, absolute_start, relative_start, duration)


class Focal:
    def __init__(self, actor, actions):
        self._actor = actor
        self._actions = actions

    @classmethod
    def from_file(cls, file, conversion=None):
        """
        conversion: Optional function that takes a string and returns a string. Used to change/merge categories.
        """
        actions: "list[Action]" = []

        rows = _read_csv(file)

        actor = _clean_value(rows[0][" Actor"])
        assert all(_clean_value(row[" Actor"]) == actor for row in rows)

        for row in rows:
            # Assume that all behaviors are mutually exclusive. Anytime we
            # see a State stop, then it is always followed by a State start.
            if row[" Event_Type"] != " State start":
                continue

            action = Action.from_row(row, conversion)
            actions.append(action)

        return cls(actor, actions)

    @classmethod
    def from_scan_file(cls, file, conversion=None):
        """
        Produces multiple "Focal" items from a single Scan file.
        """
        actions = collections.defaultdict(list)

        rows = _read_csv(file)

        for row in rows:
            # Assume that all behaviors are mutually exclusive. Anytime we
            # see a State stop, then it is always followed by a State start.
            if row[" Event_Type"] != " State start":
                continue

            action = Action.from_row(row, conversion)
            if action.name != 'Out of Sight': actions[_clean_value(row[" Actor"])].append(action)

        return [cls(actor, actions[actor]) for actor in actions]

    def relative_slice(self, start_time, stop_time):
        actions = []
        for action in self:
            # Skip actions that end before start_time
            if action.relative_stop < start_time:
                continue

            # Skip actions that start after end_time
            if action.relative_start > stop_time:
                continue

            actions.append(action.relative_slice(start_time, stop_time))  
        return self.__class__(self._actor, actions)

    def __iter__(self):
        return iter(self._actions)

    def __repr__(self):
        return f"Focal('{self._actor}', {len(self._actions)} actions)"

    
    @property
    def behaviors(self):
        return sorted(action.name for action in self._actions)

    @property
    def actor(self):
        return self._actor
