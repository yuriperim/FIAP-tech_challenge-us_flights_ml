from datetime import timedelta
import holidays


def check_group_holidays(group):
    country, state = group.name

    try:
        cal = holidays.country_holidays(country, subdiv=state)
    except Exception:
        cal = holidays.country_holidays(country)

    dates = group["DT_SCHEDULED_DEPARTURE"].dt.date
    next_dates = (group["DT_SCHEDULED_DEPARTURE"] + timedelta(days=1)).dt.date

    group["HOLIDAY"] = dates.apply(lambda x: x in cal)
    group["HOLIDAY_EVE"] = next_dates.apply(lambda x: x in cal)

    return group
