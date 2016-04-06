# -*- coding: utf-8 -*-
"""
Person class for representing a node in a genealogical tree.
"""
import re
import os
import numpy as np
import codecs
import dateutil.parser
import copy

redates = [ 
    re.compile("(?P<modifier>BEF\\. |AFT\\. |ABT\\. |EST |CAL |ABT |BEF |)"
        "(?P<day>[1-9][0-9]? )?(?P<month>[A-Z]+ )?(?P<year>[0-9]+)\\s*$"),
    re.compile(
        "(?P<modifier>BEF\\. |AFT\\. |ABT\\. |EST |CAL |ABT |BEF |)"
        "BET "
            "(?P<day1>[1-9][0-9]? )?(?P<month1>[A-Z]+ )?(?P<year1>[0-9]+)"
        " AND "
            "(?P<day2>[1-9][0-9]? )?(?P<month2>[A-Z]+ )?(?P<year2>[0-9]+)"
        "\\s*$"),
    re.compile(
        "(?P<modifier>BEF\\. |AFT\\. |ABT\\. |EST |CAL |ABT |BEF |)"
        "FROM "
            "(?P<day1>[1-9][0-9]? )?(?P<month1>[A-Z]+ )?(?P<year1>[0-9]+)"
        " TO "
            "(?P<day2>[1-9][0-9]? )?(?P<month2>[A-Z]+ )?(?P<year2>[0-9]+)"
        "\\s*$"),
    re.compile(
        "(?P<modifier>BEF\\. |AFT\\. |ABT\\. |EST |CAL |ABT |BEF |)"
        "BET\\. "
            "(?P<day1>[1-9][0-9]? )?(?P<month1>[A-Z]+ )?(?P<year1>[0-9]+)"
        " - "
            "(?P<day2>[1-9][0-9]? )?(?P<month2>[A-Z]+ )?(?P<year2>[0-9]+)"
        "\\s*$"),
]


def clean_date(datestr, print_unmatched=False):
    if datestr is None or datestr.strip() == '':
        return (None, None, None, None)
    year = None
    date = None
    year_range = None
    modifier = None
    for redate in redates:
        match = redate.match(datestr)
        if match:
            modifier = match.groupdict().get("modifier", None)
            if "year" in match.groupdict():
                year = int(match.groupdict()["year"])
                if "day" in match.groupdict() and "month" in match.groupdict() \
                        and match.groupdict()["month"] and \
                        match.groupdict()["day"]:
                    new_datestr = "%d %s %s" % (
                        year, match.groupdict()["month"],
                        match.groupdict()["day"])
                    try:
                        date = dateutil.parser.parse(new_datestr)
                    except ValueError:
                        print "Parsing the following failed:", new_datestr
                        date = None
            elif "year2" in match.groupdict():
                year_range = (int(match.groupdict()["year1"]),
                              int(match.groupdict()["year2"]))
                year = sum(year_range) / 2
            break
    else:
        if print_unmatched:
            print "no match: " + datestr
    return year, date, year_range, modifier


def clean_name(name):
    name = name.lower()
    name = re.sub('c', 'k', name)
    name = re.sub('w', 'v', name)
    name = re.sub(u'å', 'o', name)
    name = re.sub(u'[^a-zöäå ]', '', name)
    return name


class Person:
    def __init__(self, line, clean=False, blocking_key_len=1):
        self.line = line.rstrip('\n')
        L = self.line.split('\t')
        self.xref = unicode(str(L[0]), encoding='utf-8')
        #if self.xref.startswith('@'):
        #    self.xref = self.xref[1:-1]
        self.first_names = unicode(str(L[1]), encoding='utf-8').strip()
        # Handle patronymic / second name
        name_parts = self.first_names.split()
        if len(name_parts) == 0:
            self.first_name = u""
            self.patronymic = u""
            #print "Empty first name:", self.xref
        elif len(name_parts) == 1:
            self.first_name = name_parts[0]
            self.patronymic = u""
        else:
            self.first_name = name_parts[0]
            self.patronymic = u" ".join(name_parts[1:])
        self.last_name = unicode(str(L[2]), encoding='utf-8').strip()
        self.full_name = self.first_names + " " + self.last_name
        self.bdate_str = unicode(str(L[3]), encoding='utf-8')
        self.bplace = unicode(str(L[4]), encoding='utf-8')
        if clean:
            # Clean names
            self.clean_first_name = clean_name(self.first_name)
            self.clean_patronymic = clean_name(self.patronymic)
            self.clean_last_name = clean_name(self.last_name)
            # Construct a blocking key
            #self.bkey = self.clean_first_name[:blocking_key_len] + \
            #            self.clean_last_name[:blocking_key_len]
            self.bkey = self.clean_last_name[:blocking_key_len]
            # Parse date
            self.bdate_tuple = clean_date(self.bdate_str)
            self.byear, self.bdate, self.byear_range, self.bdate_modifier = \
                self.bdate_tuple
        self.mom = None
        self.dad = None
        self.kids = []

    def add_mom(self, person):
        self.mom = person.xref

    def add_dad(self, person):
        self.dad = person.xref

    def add_kid(self, person):
        # Avoid duplicate kids
        if person.xref not in self.kids:
            self.kids.append(person.xref)

    def reconstruct_line(self):
        line = u'\t'.join((self.xref, self.first_name + ' ' + self.patronymic,
                          self.last_name, self.bdate_str, self.bplace))
        self.line = line
        return line

    def __str__(self):
        ret = "%s | %s | %s | %s | %s" % (self.xref, self.first_name,
                                          self.last_name, self.bdate_str,
                                          self.bplace)
        return ret

    def copy(self):
        return copy.copy(self)


def read_people(folder=None, ind_file=None, dad_edge_file=None,
                mom_edge_file=None, clean=False, return_edges=False):
    if folder is not None:
        ind_file = os.path.join(folder, 'individuals.tsv')
        dad_edge_file = os.path.join(folder, 'dad-edges.txt')
        mom_edge_file = os.path.join(folder, 'mom-edges.txt')
    indf = codecs.open(ind_file, 'r')
    lines = indf.readlines()
    if lines[0].startswith('xref'):
        lines = lines[1:]
    people = [Person(line, clean) for line in lines]
    people_dict = {p.xref: p for p in people}
    if dad_edge_file is not None:
        dad_edges = np.loadtxt(dad_edge_file, dtype=str, delimiter=',')
        # Handle the case when there's only a single edge
        if len(dad_edges) > 0 and isinstance(dad_edges[0], basestring):
            dad_edges = [dad_edges]
        for e in dad_edges:
            parent = people_dict[e[0]]
            kid = people_dict[e[1]]
            parent.add_kid(kid)
            kid.add_dad(parent)
    if mom_edge_file is not None:
        mom_edges = np.loadtxt(mom_edge_file, dtype=str, delimiter=',')
        # Handle the case when there's only a single edge
        if len(mom_edges) > 0 and isinstance(mom_edges[0], basestring):
            mom_edges = [mom_edges]
        for e in mom_edges:
            parent = people_dict[e[0]]
            kid = people_dict[e[1]]
            parent.add_kid(kid)
            kid.add_mom(parent)
    if return_edges:
        #edges = np.vstack((mom_edges, dad_edges))
        return people, people_dict, list(mom_edges), list(dad_edges)
    return people, people_dict
