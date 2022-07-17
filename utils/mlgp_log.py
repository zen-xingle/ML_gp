

class mlgp_log:
    def __init__(self) -> None:
        self.log_level = 1

    @staticmethod
    def error(*args):
        meg = ' '.join(args)
        print('\033[1;41m' + 'ERROR: ' + meg + '\033[0m')

    @staticmethod
    def e(*args):
        meg = ' '.join(args)
        print('\033[1;41m' + 'ERROR: ' + meg + '\033[0m')   

    @staticmethod
    def info(*args):
        meg = ' '.join(args)
        print('\033[1;40m' + 'INFO: ' + meg + '\033[0m')

    def i(*args):
        meg = ' '.join(args)
        print('\033[1;40m' + 'INFO: ' + meg + '\033[0m')

    @staticmethod
    def warning(*args):
        meg = ' '.join(args)
        print('\033[1;33m' + 'WARNING: ' + meg + '\033[0m')

    def w(*args):
        meg = ' '.join(args)
        print('\033[1;33m' + 'WARNING: ' + meg + '\033[0m')

    @staticmethod
    def debug(*args):
        meg = ' '.join(args)
        print('\033[1;45m' + 'DEBUG: ' + meg + '\033[0m')

    def d(*args):
        meg = ' '.join(args)
        print('\033[1;45m' + 'DEBUG: ' + meg + '\033[0m')


if __name__ == '__main__':
    mlgp_log.i('This is info')
    mlgp_log.w('This is warning')
    mlgp_log.e('This is error')
    mlgp_log.d('This is debug')