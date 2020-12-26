import os


class Files:
    LAD_PATH = os.path.join(
        os.getcwd(),
        "covid19/data/Local_Authority_Districts__December_2019__Boundaries_UK_BFC.shp",
    )
    REG_PATH = os.path.join(
        os.getcwd(),
        "covid19/data/Ward_to_Local_Authority_District_to_County_to_Region_to_Country_(December_2019)_Lookup_in_United_Kingdom.csv",
    )
    POP_PATH = os.path.join(
        os.getcwd(),
        "covid19/data/population.csv",
    )
    CASES_PATH = os.path.join(
        os.getcwd(),
        "covid19/data/cases.csv",
    )
    UTLA_PATH = os.path.join(
        os.getcwd(),
        "covid19/data/Lower_Tier_Local_Authority_to_Upper_Tier_Local_Authority_(April_2019)_Lookup_in_England_and_Wales.csv",
    )


class API:
    all_data = {
        "date": "date",
        "areaName": "areaName",
        "areaCode": "areaCode",
        "newCasesByPublishDate": "newCasesByPublishDate",
        "cumCasesByPublishDate": "cumCasesByPublishDate",
        "newCasesBySpecimenDate": "newCasesBySpecimenDate",
        "cumCasesBySpecimenDateRate": "cumCasesBySpecimenDateRate",
        "cumCasesBySpecimenDate": "cumCasesBySpecimenDate",
        "newDeathsByDeathDate": "newDeathsByDeathDate",
        "cumDeathsByDeathDate": "cumDeathsByDeathDate",
        "maleCases": "maleCases",
        "femaleCases": "femaleCases",
        "newPillarOneTestsByPublishDate": "newPillarOneTestsByPublishDate",
        "cumPillarOneTestsByPublishDate": "cumPillarOneTestsByPublishDate",
        "newPillarTwoTestsByPublishDate": "newPillarTwoTestsByPublishDate",
        "cumPillarTwoTestsByPublishDate": "cumPillarTwoTestsByPublishDate",
        "newPillarThreeTestsByPublishDate": "newPillarThreeTestsByPublishDate",
        "cumPillarThreeTestsByPublishDate": "cumPillarThreeTestsByPublishDate",
        "newPillarFourTestsByPublishDate": "newPillarFourTestsByPublishDate",
        "cumPillarFourTestsByPublishDate": "cumPillarFourTestsByPublishDate",
        "newAdmissions": "newAdmissions",
        "cumAdmissions": "cumAdmissions",
        "cumAdmissionsByAge": "cumAdmissionsByAge",
        "cumTestsByPublishDate": "cumTestsByPublishDate",
        "newTestsByPublishDate": "newTestsByPublishDate",
        "covidOccupiedMVBeds": "covidOccupiedMVBeds",
        "hospitalCases": "hospitalCases",
        "plannedCapacityByPublishDate": "plannedCapacityByPublishDate",
        "newDeaths28DaysByPublishDate": "newDeaths28DaysByPublishDate",
        "cumDeaths28DaysByPublishDate": "cumDeaths28DaysByPublishDate",
        "cumDeaths28DaysByPublishDateRate": "cumDeaths28DaysByPublishDateRate",
        "newDeaths28DaysByDeathDate": "newDeaths28DaysByDeathDate",
        "cumDeaths28DaysByDeathDate": "cumDeaths28DaysByDeathDate",
        "cumDeaths28DaysByDeathDateRate": "cumDeaths28DaysByDeathDateRate",
    }

    cases_and_deaths = {
        "date": "date",
        "areaName": "areaName",
        "areaCode": "areaCode",
        "newCasesByPublishDate": "newCasesByPublishDate",
        "cumCasesByPublishDate": "cumCasesByPublishDate",
        "newDeathsByDeathDate": "newDeathsByDeathDate",
        "cumDeathsByDeathDate": "cumDeathsByDeathDate",
    }
