def time_convert(str_time):
    """
    This gives time in milisecond

    Arguments
    ---------
    str_time : str
        The time in string type
        ex) 10:09:20.123

    str_time_format : str
    The time format in string type
    ex) "%H:%M:%S.%f"

    Returns
    -------
    float
        A milisecond time

    """    
    hh, mm, sec_mili = str_time.split(':')
    total_milis = float(hh) * 60 * 60 + float(mm) * 60 + float(sec_mili)
                
    return total_milis