class VggCfg(object):
    benchmark = dict(agri_urban=[256, 'W', 'M3', 512, 'W', 'M3', 1024, 2048, 'E'],
                     agri_forest=[128, 'W', 'M3', 256, 'W', 'M3', 512, 1024, 'E'],
                     forest_agri=[128, 'W', 'M3', 256, 'W', 'M3', 512, 1024, 'E'],
                     )


class ResCfg(object):
    temp = dict(m1=['B/64/1', 'B/64/1', 'B/128/1', 'B/128/3', 'B/256/1', 'B/256/1', 'B/512/1', 'B/512/1', 'E'],
                m2=['B/128/1', 'B/128/1', 'B/256/1', 'B/256/3', 'B/512/1', 'B/512/1', 'B/1024/1', 'B/1024/1', 'E'],
                m3=['N/128/1', 'N/128/1', 'N/256/1', 'N/256/3', 'N/512/1', 'N/512/1', 'N/1024/1', 'N/1024/1', 'E']
                )


class AECfg(object):
    basic = dict(m1=['32/3', 'M3', '64/3', 'M3'],)


class ClfCfg(object):
    ae = dict(agri_urban=[7 * 81 + 12, 7 * 81 + 12, 7 * 81 + 12, 100],
              forest_agri=[7 * 81 + 12, 7 * 81 + 12, 7 * 81 + 12],
              agri_forest=[7 * 81 + 12, 7 * 81 + 12, 7 * 81 + 12])

    conv = dict(agri_urban=[2048 + 12, 800, 300, 120, 60],
                forest_agri=[1024 + 12, 400, 80, 7],
                agri_forest=[1024 + 12, 400, 80, 7]
                )
