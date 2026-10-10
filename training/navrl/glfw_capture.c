/* Capture the native Simulate window; all other GLFW symbols use the real library. */
#define _GNU_SOURCE
#include <dlfcn.h>
static void *window;
void *glfwCreateWindow(int width,int height,const char *title,void *monitor,void *share) {
  typedef void *(*Create)(int,int,const char *,void *,void *);
  Create create=(Create)dlsym(RTLD_NEXT,"glfwCreateWindow");
  window=create(width,height,title,monitor,share);
  return window;
}
void *navigation_glfw_window(void) {return window;}
