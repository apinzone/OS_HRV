from sqlite3 import Time
from TimeDomainHRV import TimeDomainHRV


def main():
    myHRV = TimeDomainHRV()
    myHRV.run()

if __name__ == "__main__":
    main()