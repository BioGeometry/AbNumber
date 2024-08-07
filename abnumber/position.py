import copy
from typing import List, Union

from abnumber.common import _validate_chain_type, _chain_type_to_prefix, SCHEME_POSITION_TO_REGION, SCHEME_VERNIER, SCHEME_INTERFACE, SCHEME_NANOBODY, POS_REGEX, _cdr_definition_to_scheme


class Position:
    """Numbered position using a given numbering scheme

    Used as a key to store Position -> Amino acid information.

    Position objects are sortable according to the schema simply using ``sorted()``.
    """
    def __init__(self, chain_type: str, number: int, letter: str, scheme: str):
        _validate_chain_type(chain_type, scheme)
        self.chain_type: str = chain_type
        self.number: int = int(number)
        self.letter: str = letter.strip()
        self.scheme: str = scheme
        self.cdr_definition: str = self.scheme
        self.cdr_definition_position: int = self.number

    def copy(self):
        return copy.copy(self)

    def _key(self):
        # Note: We are not including chain_type, but just Heavy/Light flag, to keep Kappa and Lambda chain positions equal
        return self.chain_type_prefix(), self.number, self.letter, self.scheme

    def __repr__(self):
        return f'{self.chain_type_prefix()}{self.number}{self.letter} ({self.scheme})'

    def __str__(self):
        return self.format()

    def set_cdr_definition(self, cdr_definition: str, cdr_definition_position: int):
        assert cdr_definition is not None, 'cdr_definition is required'
        assert cdr_definition_position is not None, 'cdr_definition_position is required'
        _validate_chain_type(self.chain_type, _cdr_definition_to_scheme(cdr_definition))
        self.cdr_definition = cdr_definition
        self.cdr_definition_position = cdr_definition_position

    def format(self, chain_type=True, region=False, rjust=False, ljust=False, fillchar=' '):
        """Format Position to string

        :param chain_type: Add chain type prefix (H/L)
        :param region: Add region prefix (FR1, CDR1, ...)
        :param rjust: Align text to the right
        :param ljust: Align text to the left
        :param fillchar: Characer to use for alignment padding
        :return: formatted string
        """
        formatted = f'{self.number}{self.letter}'
        if chain_type:
            formatted = f'{self.chain_type_prefix()}{formatted}'
        if region:
            formatted = f'{self.get_region()} {formatted}'
        just = 4 + 1* int(chain_type) + 5 * int(region)
        if rjust:
            formatted = formatted.rjust(just, fillchar)
        if ljust:
            formatted = formatted.ljust(just, fillchar)
        return formatted

    def __hash__(self):
        return self._key().__hash__()

    def __eq__(self, other):
        return isinstance(other, Position) and self._key() == other._key()

    def __ge__(self, other):
        return self == other or self > other

    def __le__(self, other):
        return self == other or self < other

    def __lt__(self, other):
        if not isinstance(other, Position):
            raise TypeError(f'Cannot compare Position object with {type(other)}: {other}')
        assert self.is_heavy_chain() == other.is_heavy_chain(), f'Positions do not come from the same chain: {self}, {other}'
        assert self.scheme == other.scheme, 'Comparing positions in different schemes is not implemented'
        return self._sort_key() < other._sort_key()

    def chain_type_prefix(self):
        return _chain_type_to_prefix(self.chain_type)

    def _sort_key(self):
        letter_ord = ord(self.letter) if self.letter else 0
        if self.scheme == 'imgt':
            if self.number in [33, 61, 112]:
                # position 112 is sorted in reverse
                letter_ord = -letter_ord
        elif self.scheme in ['chothia', 'kabat', 'aho']:
            # all letters are sorted alphabetically for these schemes
            pass
        else:
            raise NotImplementedError(f'Cannot compare positions of scheme: {self.scheme}')
        return self.is_heavy_chain(), self.number, letter_ord

    def get_region(self, show_vernier=False, show_interface=False, show_nanobody_conserved_spots=False):
        """Get string name of this position's region.
        If `show_vernier` is True, the region name will be extended with " Vernier" if the position is in the vernier zone.
        If `show_interface` is True, the region name will be extended with " Interface" if the position is on the VH-VL interface.

        :return: uppercase string, one of: ``"FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"``
        """
        if self.cdr_definition in SCHEME_POSITION_TO_REGION:
            regions = SCHEME_POSITION_TO_REGION[self.cdr_definition]
        else:
            regions = SCHEME_POSITION_TO_REGION[f'{self.cdr_definition}_{self.chain_type}']
        region = regions[self.cdr_definition_position]
        if show_vernier and self.is_in_vernier():
            region += ' Vernier'
        if show_interface and self.is_in_interface():
            region += ' Interface'
        if show_nanobody_conserved_spots and self.is_in_nanobody_conserved_spots():
            region += ' Nanobody'
        return region

    def is_in_cdr(self):
        """Check if given position is found in the CDR regions"""
        return self.get_region().lower().startswith('cdr')

    def is_in_vernier(self):
        key = f'{self.cdr_definition}_{self.chain_type}'
        if key not in SCHEME_VERNIER:
            raise NotImplementedError(f'Vernier zone not implemented for {key}')
        return self.cdr_definition_position in SCHEME_VERNIER.get(key, [])

    def is_in_interface(self):
        """Check if given position is found in the VH-VL interface"""
        key = f'{self.cdr_definition}_{self.chain_type}'
        if key not in SCHEME_INTERFACE:
            raise NotImplementedError(f'VH-VL interface not implemented for {key}')
        return self.cdr_definition_position in SCHEME_INTERFACE.get(key, [])

    def is_in_nanobody_conserved_spots(self):
        """Check if given position is found in the Nanobody conserved spots, which are reported to be important for stability and solubility of nanobodies"""
        if self.chain_type != 'H':
            return False

        key = f'{self.cdr_definition}_{self.chain_type}'
        if key not in SCHEME_NANOBODY:
            raise NotImplementedError(f'Nanobody conserved spots not implemented for {key}')
        return self.cdr_definition_position in SCHEME_NANOBODY.get(key, [])

    @classmethod
    def from_string(cls, position, chain_type, scheme):
        """Create Position object from string, e.g. "H5"

        Note that Positions parsed from string do not support separate CDR definitions.
        """
        match = POS_REGEX.match(position.upper())
        _validate_chain_type(chain_type, scheme)
        expected_chain_prefix = _chain_type_to_prefix(chain_type)
        if match is None:
            raise IndexError(f'Expected position format chainNumberLetter '
                             f'(e.g. "{expected_chain_prefix}112A" or "112A"), got: "{position}"')
        chain_prefix, number, letter = match.groups()
        number = int(number)
        if chain_prefix and expected_chain_prefix != chain_prefix:
            raise IndexError(f'Use no prefix or "{expected_chain_prefix}" prefix for "{chain_type}" chain. '
                             f'Got: "{chain_prefix}".')
        return cls(chain_type=chain_type, number=number, letter=letter, scheme=scheme)

    def is_heavy_chain(self):
        return self.chain_type == 'H'

    def is_light_chain(self):
        return self.chain_type in 'KL'

    def is_alpha_chain(self):
        return self.chain_type == 'A'

    def is_beta_chain(self):
        return self.chain_type == 'B'


def sort_positions(positions: List[str], chain_type: str, scheme: str) -> List:
    """Sort position strings to correct order based on given scheme"""
    has_prefix = [p[0] in 'HLAB' for p in positions]
    assert all(has_prefix) or not any(has_prefix), 'Inconsistent position prefix'
    has_prefix = all(has_prefix)

    position_objects = [Position.from_string(p, chain_type=chain_type, scheme=scheme) for p in positions]

    return [p.format(chain_type=has_prefix) for p in sorted(position_objects)]
