"""Regression cases for the user's revised scoring rule; no simulations."""
import unittest
from rejudge_health_records import verdict, counts, rows_in


class HealthVerdictTest(unittest.TestCase):
    def test_user_examples_and_reversed_seats(self):
        for own, foe in ((.8, .1), (.015, 0), (.03, 0)):
            self.assertEqual(verdict(dict(own_health=own, opp_health=foe)), 'win')
            self.assertEqual(verdict(dict(own_health=foe, opp_health=own)), 'loss')

    def test_rounded_ties_are_draws(self):
        for own, foe in ((0, 0), (.5, .5), (.50004, .5), (.00004, 0)):
            self.assertEqual(verdict(dict(own_health=own, opp_health=foe)), 'draw')
        self.assertEqual(verdict(dict(own_health=.5001, opp_health=.5)), 'win')

    def test_invalid_health_is_not_a_draw(self):
        for own in (float('nan'), float('inf'), -.1, 1.1):
            self.assertEqual(verdict(dict(own_health=own, opp_health=0)), 'invalid')

    def test_episode_extraction_excludes_aggregate_health(self):
        data = {'summary': {'own_health': .5, 'opp_health': .2},
                'episodes': [dict(own_health=.8, opp_health=.1, outcome='timeout', seed=1),
                             dict(own_health=.015, opp_health=0, outcome='mutual', seed=2),
                             dict(own_health=.5, opp_health=.5, outcome='timeout', seed=3)]}
        result = counts(list(rows_in(data)))
        self.assertEqual(result['records'], 3)
        self.assertEqual(result['draw_to_win'], 2)
        self.assertEqual(result['draw_to_draw'], 1)


if __name__ == '__main__':
    unittest.main()
