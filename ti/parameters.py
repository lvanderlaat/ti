#!/usr/bin/env python


"""
"""


# Python Standard Library
import os

# Other dependencies
import numpy as np
import pandas as pd

# Local files


__author__ = 'Leonardo van der Laat'
__email__ = 'laat@umich.edu'


def load():
    folderpath = os.path.split(os.path.abspath(__file__))[0]
    df = pd.read_csv(os.path.join(folderpath, 'units.csv'))
    return df


def as_dict():
    return load().set_index('key').to_dict(orient='index')


def exp_fmt(v, fmt):
    return f'$\\num{{{v:{fmt}}}}$'


def to_latex(c):
    df = load()
    df = df[~df.key.isin(['S', 'max_freq', 'tau'])]

    values = []
    for _, row in df.iterrows():
        unit = f'({row.unit})'
        if unit == '(nan)':
            unit = ''

        key = row.key
        v = c.model[key]

        conversion = row.conversion
        if np.isnan(conversion):
            conversion = 1

        fmt = row.fmt
        if fmt[-1] == 'g':
            fmt.replace('g', 'f')

        if type(v) == list:
            # v0 = f'{v[0]*conversion}'
            v0 = v[0]*conversion
            v1 = v[1]*conversion

            if row.type == 'int':
                v0 = int(v0)
                v1 = int(v1)
            v0_str = f'{v0:{fmt}}'
            v1_str = f'{v1:{fmt}}'

            if row.fmt[-1] == 'g':
                v0_str = exp_fmt(v0, fmt)
                v1_str = exp_fmt(v1, fmt)

            value = f'[{v0_str} — {v1_str}]'
        else:
            v *= conversion
            if row.type == 'int':
                v = int(v)
            value = f'{v:{fmt}}'
            if row.fmt[-1] == 'g':
                value = exp_fmt(v, fmt)

        values.append(value)

    df['Parameter'] = df.math_notation
    df['Description'] = df.description
    df['Unit'] = df.unit
    df['Value'] = values
    df = df['Parameter Description Value Unit'.split()]

    latex_code = df.style.hide(axis='index').to_latex(
        position='!htb',
        position_float='centering',
        hrules=True,
        label='tab:tremor_simulation',
        caption='Tremor simulation parameters',
        siunitx=True,
    )
    for _rule in 'toprule midrule bottomrule'.split():
        latex_code = latex_code.replace(_rule, 'hline')
    return latex_code


def write_latex(c, folderpath):
    with open(os.path.join(folderpath, 'params.tex'), 'w') as f:
        f.write(to_latex(c))


def get_parameter_label(key):
    _p = as_dict()[key]
    unit = _p['unit'].replace('\\', '')
    label = _p['math_notation']
    if unit != ' ':
        label += f' [{unit}]'
    return label


if __name__ == '__main__':
    import config
    c = config.read('./example/config.yml')
    write_latex(c, './')
