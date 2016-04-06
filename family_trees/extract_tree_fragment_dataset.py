"""
Given a large genealogical tree, extract k random subgraphs of it.

See: get_k_fragments() below.
"""
import numpy as np
import codecs
import os
import person
import random


def pick_neighbor(p, forbidden_set=set()):
    options = []
    if p.mom is not None and p.mom not in forbidden_set:
        options.append(p.mom)
    if p.dad is not None and p.dad not in forbidden_set:
        options.append(p.dad)
    for kid in p.kids:
        if kid not in forbidden_set:
            options.append(kid)
    if len(options) == 0:
        return None
    return random.choice(options)


def extract_fragment(people, people_dict, n_people=100, seed_person=None):
    if seed_person is None:
        current = random.choice(people)
    else:
        current = seed_person
    frag = set()
    # Number of steps for which no new person is encountered
    n_no_new = 0
    while len(frag) < n_people:
        neighbor_found = False
        while not neighbor_found:
            next_xref = pick_neighbor(current)
            if next_xref is None:
                current = random.choice(people)
            else:
                neighbor_found = True
        if next_xref not in frag:
            frag.add(next_xref)
            n_no_new = 0
        else:
            n_no_new += 1
        if n_no_new > n_people * 10:
            # Random walked for too long without finding anything, reset to a
            # random node
            n_no_new = 0
            next_xref = random.choice(people).xref
        current = people_dict[next_xref]
    frag = list(frag)
    frag_people = []
    for xref in frag:
        frag_people.append(people_dict[xref].copy())
    return frag_people


def write_fragment(frag, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    indf = codecs.open(os.path.join(folder, 'individuals.tsv'), 'w',
                       encoding='utf-8')
    momf = codecs.open(os.path.join(folder, 'mom-edges.txt'), 'w',
                       encoding='utf-8')
    dadf = codecs.open(os.path.join(folder, 'dad-edges.txt'), 'w',
                       encoding='utf-8')
    frag_xrefs = [p.xref for p in frag]
    for p in frag:
        indf.write(p.line + '\n')
        if p.mom in frag_xrefs:
            momf.write('%s,%s\n' % (p.mom, p.xref))
        if p.dad in frag_xrefs:
            dadf.write('%s,%s\n' % (p.dad, p.xref))
    print "Wrote to:", folder


def read_name_alternatives(path):
    """
    Add noise to fragments.
    :param path: File path to the alternative names file.
    :return: Map from name to the list of alternative names.
    """
    alt_map = {}
    lines = codecs.open(path).readlines()
    for l in lines:
        l = unicode(l, encoding='iso-8859-1')
        names = [name for name in l.strip().split('@')]
        for name in names:
            alt_map[person.clean_name(name)] = names
    return alt_map


def add_noise(ppeople, p_round_byear=1, year_bin_width=10):
    folder = os.path.dirname(__file__)
    f_alts = read_name_alternatives(os.path.join(folder, 'data/etunimi.cnv'))
    l_alts = read_name_alternatives(os.path.join(folder, 'data/sukunimi.cnv'))
    p_alts = read_name_alternatives(os.path.join(folder, 'data/patron.cnv'))
    lname_map = {}
    n_mod_f = 0
    n_mod_p = 0
    n_mod_l = 0
    for p in ppeople:
        if p.clean_first_name in f_alts and len(f_alts[p.clean_first_name]) > 1:
            alts = f_alts[p.clean_first_name]
            p.first_name = alts[np.random.randint(len(alts))]
            p.clean_first_name = person.clean_name(p.first_name)
            n_mod_f += 1

        if p.clean_patronymic in p_alts and len(p_alts[p.clean_patronymic]) > 1:
            alts = p_alts[p.clean_patronymic]
            p.patronymic = alts[np.random.randint(len(alts))]
            p.clean_patronymic = person.clean_name(p.patronymic)
            n_mod_p += 1

        if p.clean_last_name in lname_map and \
                p.last_name != lname_map[p.clean_last_name]:
            n_mod_l += 1
        elif p.clean_last_name in l_alts and len(l_alts[p.clean_last_name]) > 1:
            alts = l_alts[p.clean_last_name]
            new_lname = alts[np.random.randint(len(alts))]
            lname_map[p.clean_last_name] = new_lname
            n_mod_l += 1
        else:
            lname_map[p.clean_last_name] = p.last_name
        p.last_name = lname_map[p.clean_last_name]
        p.clean_last_name = person.clean_name(p.last_name)

        if p.byear is not None:
            if random.random() < p_round_byear:
                p.byear = int(np.round(p.byear / year_bin_width) *
                              year_bin_width)
            p.bdate_str = str(int(p.byear))
        p.reconstruct_line()
    print "Modified %d first, %d lastnames, and %d patronymics (out of %d)." % \
            (n_mod_f, n_mod_l, n_mod_p, len(ppeople))


def get_k_fragments(k, n_people=1000, label='', check_if_exists=True):
    """
    Create k family tree fragments.
    :param k: Number of trees.
    :param n_people: Number of people per tree.
    :param label: label used in the file name
    :param check_if_exists: Whether to use existing data if available or not.
    :return: list of file names
    """
    folder = os.path.dirname(__file__)
    fname = os.path.join(folder, 'data/orig/')
    people, people_dict = person.read_people(fname, clean=True)

    seed_person = random.choice(people)
    fnames = []
    for i in range(k):
        fname = os.path.join(
            folder, "data/rand_frag_people{}_label{}_{}/".format(n_people,
                                                                 label, i))
        if not check_if_exists or not os.path.exists(fname):
            frag = extract_fragment(people, people_dict, n_people, seed_person)
            seed_person = random.choice(frag)
            add_noise(frag, p_round_byear=1, year_bin_width=10)
            write_fragment(frag, fname)
        fnames.append(fname)
    return fnames


def main():
    fnames = get_k_fragments(5, 100)
    print fnames

if __name__ == "__main__":
    main()