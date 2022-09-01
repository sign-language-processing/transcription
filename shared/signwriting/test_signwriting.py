import unittest

from .signwriting import fsw_to_sign, join_signs


class ParseSignCase(unittest.TestCase):

    def test_get_box(self):
        fsw = 'M123x456S1f720487x492'
        sign = fsw_to_sign(fsw)
        self.assertEqual(sign["box"]["symbol"], "M")
        self.assertEqual(sign["box"]["position"], (123, 456))


class JoinSignsCase(unittest.TestCase):

    def test_join_two_characters(self):
        char_a = 'M507x507S1f720487x492'
        char_b = 'M507x507S14720493x485'
        result_sign = join_signs(char_a, char_b)
        self.assertEqual(result_sign, 'M500x500S1f720487x493S14720493x508')

    def test_join_alphabet_characters(self):
        chars = [
            "M510x508S1f720490x493", "M507x511S14720493x489", "M509x510S16d20492x490", "M508x515S10120492x485",
            "M508x508S14a20493x493", "M511x515S1ce20489x485", "M515x508S1f000486x493", "M515x508S11502485x493",
            "M511x510S19220490x491", "M519x518S19220498x499S2a20c482x483"
        ]
        result_sign = join_signs(*chars, spacing=10)
        # pylint: disable=line-too-long
        self.assertEqual(
            result_sign,
            'M500x500S1f720490x362S14720493x387S16d20492x419S10120492x449S14a20493x489S1ce20489x514S1f000486x554S11502485x579S19220490x604S19220498x649S2a20c482x633'  # noqa: E501
        )


if __name__ == '__main__':
    unittest.main()
