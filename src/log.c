#include "log.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

static int g_level = RB_LOG_INFO;

void rb_log_set_level(int level)
{
    g_level = level;
}

int rb_log_get_level(void)
{
    return g_level;
}

void rb_log(int level, const char *fmt, ...)
{
    if (level > g_level)
        return;

    va_list ap;
    va_start(ap, fmt);
    vfprintf((level <= RB_LOG_WARN) ? stderr : stdout, fmt, ap);
    va_end(ap);
}

int rb_log_level_from_string(const char *s)
{
    if (strcmp(s, "error") == 0)
        return RB_LOG_ERROR;
    if (strcmp(s, "warn") == 0 || strcmp(s, "warning") == 0)
        return RB_LOG_WARN;
    if (strcmp(s, "info") == 0)
        return RB_LOG_INFO;
    if (strcmp(s, "debug") == 0)
        return RB_LOG_DEBUG;
    return -1;
}
