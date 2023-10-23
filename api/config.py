class Config:
    pass

class DevelopmentConfig(Config):
    DEBUG = False

class LocalConfig(Config):
    DEBUG = True

config = {
    'development': DevelopmentConfig,
    'local': LocalConfig,
}