/* Test-only libtcc double: inject recoverable relocation denial without public
   failure switches or changing machine-wide security policy. */
#include <stdlib.h>

typedef struct {
    void *opaque;
    void (*error)(void *, const char *);
} TCCState;

TCCState *tcc_new(void) { return calloc(1, sizeof(TCCState)); }
void tcc_delete(TCCState *s) { free(s); }
void tcc_set_error_func(TCCState *s, void *opaque, void (*error)(void *, const char *)) {
    s->opaque = opaque;
    s->error = error;
}
int tcc_set_output_type(TCCState *s, int type) { (void)s; (void)type; return 0; }
int tcc_compile_string(TCCState *s, const char *source) { (void)s; (void)source; return 0; }
int tcc_add_symbol(TCCState *s, const char *name, const void *value) {
    (void)s; (void)name; (void)value; return 0;
}
int tcc_relocate(TCCState *s) {
    if (s->error)
        s->error(s->opaque, "tccrun: executable mapping failed: Permission denied");
    return -1;
}
void *tcc_get_symbol(TCCState *s, const char *name) { (void)s; (void)name; return NULL; }
