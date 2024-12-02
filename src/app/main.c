#include "logging.h"
#include "rtc.h"
#include "system.h"

#include "ai.h"

int main(void) {
   setup_hardware();

   print("Initializing peripherals...\n");
   rtc_init();
   rtc_set_time_to_compile_time();
   system_enable_interrupts(true);
   print("All peripherals initialized!\n");

   float buf[8000];
   for (unsigned i = 0; i < sizeof(buf) / sizeof(*buf); ++i) buf[i] = 0.0;
   float embed[16];
   preprocess_and_encode(buf, sizeof(buf) / sizeof(*buf), 8000, embed);
   for (unsigned i = 0; i < sizeof(embed) / sizeof(*embed); ++i) {
      print(" -> ", i);
   }

   while (true) {
      print("Going to sleep...\n");
      system_enter_deep_sleep_mode();
   }

   system_reset();
   return 0;
}
