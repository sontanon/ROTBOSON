// ROTBOSON logging.
//
// A tiny level-gated logger that replaces the ad-hoc "*** banner" printf spam.
// Levels (ascending verbosity):
//   RB_LOG_ERROR  -- fatal/error messages (stderr)
//   RB_LOG_WARN   -- warnings, non-fatal diagnostics (stderr)
//   RB_LOG_INFO   -- normal progress/banner output (stdout)
//   RB_LOG_DEBUG  -- developer diagnostics (stdout)
//
// The active level is set once from the "loglevel" parameter (parser.c) and
// defaults to RB_LOG_INFO so the pre-Phase-5 stdout is preserved unless the
// user asks for quieter output.
#ifndef ROTBOSON_LOG_H
#define ROTBOSON_LOG_H

#define RB_LOG_ERROR 0
#define RB_LOG_WARN 1
#define RB_LOG_INFO 2
#define RB_LOG_DEBUG 3

// Set / query the global log level.
void rb_log_set_level(int level);
int rb_log_get_level(void);

// Emit a message at `level`. Messages at levels above the configured level
// are suppressed. WARN and ERROR go to stderr; INFO and DEBUG go to stdout.
void rb_log(int level, const char *fmt, ...);

// Parse a textual log level ("error", "warn", "info", "debug"); returns -1
// if unknown. Used by the parser for the "loglevel" key.
int rb_log_level_from_string(const char *s);

#endif /* ROTBOSON_LOG_H */
