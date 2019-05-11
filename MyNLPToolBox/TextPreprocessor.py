def remove_unicode(myStr):
    """
    :param myStr:  any strings would work
    :return: a string with unicode-characters removed (e.g:\u00a9, \u201d)
    """
    return myStr.encode('ascii', 'ignore').decode('unicode-escape')


def remove_special(myStr):
    """
    :param myStr:  any strings would work
    :return: a string with special-characters removed
    """
    return ''.join(char for char in myStr if char.isalnum() or char == ' ' or char == '-')


def lowercase(myStr):
    """
    :param myStr:  any strings would work
    :return: a string with all of its character decapitalized
    """
    return myStr.lower()


def remove_accents(myStr):
    """
    :param myStr: any strings would work
    :return: the string with accents removed
    """
    s1 = u'ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚÝàáâãèéêìíòóôõùúýĂăĐđĨĩŨũƠơƯưẠạẢảẤấẦầẨẩẪẫẬậẮắẰằẲẳẴẵẶặ' \
         u'ẸẹẺẻẼẽẾếỀềỂểỄễỆệỈỉỊịỌọỎỏỐốỒồỔổỖỗỘộỚớỜờỞởỠỡỢợỤụỦủỨứỪừỬửỮữỰựỲỳỴỵỶỷỸỹ'
    s0 = u'AAAAEEEIIOOOOUUYaaaaeeeiioooouuyAaDdIiUuOoUuAaAaAaAaAaAaAaAaAaAaAaAa' \
         u'EeEeEeEeEeEeEeEeIiIiOoOoOoOoOoOoOoOoOoOoOoOoUuUuUuUuUuUuUuYyYyYyYy'
    s = ''
    for c in myStr:
        if c in s1:
            s += s0[s1.index(c)]
        else:
            s += c
    return s


def process(df, modes=('remove_unicode', 'remove_special', 'lowercase', 'remove_accents', 'remove_stopwords')):
    """
    :param df: any dataframes would work
    :param modes: list of modes name (string-typed) you want to use
    :return:
    """
    for mode in modes:
        df.loc[:, 'headline']=df['headline'].apply(globals()[mode])
