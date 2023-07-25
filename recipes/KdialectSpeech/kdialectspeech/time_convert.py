def time_convert(str_time:str) -> float:
    """
    This gives time in milisecond

    Arguments
    ---------
    str_time format : "%H:%M:%S.%f"
        ex) 10:09:20.123

    Returns
    -------
    A milisecond time
    """    
    hh, mm, sec_mili = str_time.split(':')
    total_milis = float(hh) * 60 * 60 + float(mm) * 60 + float(sec_mili)
                
    return total_milis