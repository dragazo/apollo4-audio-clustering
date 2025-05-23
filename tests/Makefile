PROJECT := Apollo4StarterCode
CONFIG := bin
SHELL := /bin/bash

ifdef BOARD_REV
REVISION := $(BOARD_REV)
else
REVISION := A
endif

BSP = apollo4_pro
PART = apollo4p
PART_DEF = AM_PART_APOLLO4P

$(info Building test for Revision $(REVISION):)
$(info BSP = $(BSP))
$(info PART = $(PART))
$(info PART_DEF = $(PART_DEF))

TOOLCHAIN ?= arm-none-eabi
CPU = cortex-m4
FPU = fpv4-sp-d16
FABI = hard
FLASH_START = 0x00018000
ID_FLASH_LOCATION = 0x001FFFF8

DEFINES  = -D_HW_REVISION=$(REVISION)
DEFINES += -D_DATETIME="\"$(shell date -u)\""
DEFINES += -DPART_$(PART)
DEFINES += -D$(PART_DEF)
DEFINES += -D'SEGGER_RTT_SECTION="SHARED_RW"'
DEFINES += -DAM_PACKAGE_BGA
DEFINES += -DAM_DEBUG_PRINTF
DEFINES += -Dgcc

LINKER_FILE := ../AmbiqSDK/bsp/$(BSP)/linker/a3em.ld
STARTUP_FILE := ../AmbiqSDK/bsp/$(BSP)/linker/startup_gcc.c

#### Required Executables ####
CC = $(TOOLCHAIN)-gcc
CCPP = $(TOOLCHAIN)-g++
GCC = $(TOOLCHAIN)-gcc
CPP = $(TOOLCHAIN)-cpp
LD = $(TOOLCHAIN)-ld
CP = $(TOOLCHAIN)-objcopy
OD = $(TOOLCHAIN)-objdump
RD = $(TOOLCHAIN)-readelf
AR = $(TOOLCHAIN)-ar
SIZE = $(TOOLCHAIN)-size
RM = $(shell which rm 2>/dev/null)
EXECUTABLES = CC LD CP OD AR RD SIZE GCC
K := $(foreach exec,$(EXECUTABLES),\
        $(if $(shell which $($(exec)) 2>/dev/null),,\
        $(info $(exec) not found on PATH ($($(exec))).)$(exec)))
$(if $(strip $(value K)),$(info Required Program(s) $(strip $(value K)) not found))

ifneq ($(strip $(value K)),)
all clean:
	$(info Tools $(TOOLCHAIN)-gcc not installed.)
	$(RM) -rf bin
else

INCLUDES  = -I../AmbiqSDK/bsp/$(BSP)
INCLUDES += -I../AmbiqSDK/mcu/$(PART)
INCLUDES += -I../AmbiqSDK/mcu/$(PART)/hal
INCLUDES += -I../AmbiqSDK/mcu/$(PART)/hal/mcu
INCLUDES += -I../AmbiqSDK/CMSIS/AmbiqMicro/Include
INCLUDES += -I../AmbiqSDK/CMSIS/ARM/Include
INCLUDES += -I../AmbiqSDK/devices
INCLUDES += -I../AmbiqSDK/utils
INCLUDES += -I../src/app
INCLUDES += -I../src/boards
INCLUDES += -I../src/boards/rev$(REVISION)
INCLUDES += -I../src/external/fatfs
INCLUDES += -I../src/external/segger
INCLUDES += -I../src/peripherals/include
INCLUDES += -I./peripherals

VPATH  = ../AmbiqSDK/bsp/$(BSP)/linker
VPATH += ../AmbiqSDK/mcu/$(PART)/hal
VPATH += ../AmbiqSDK/mcu/$(PART)/hal/mcu
VPATH += ../AmbiqSDK/devices
VPATH += ../AmbiqSDK/utils
VPATH += ../src/app
VPATH += ../src/boards
VPATH += ../src/boards/rev$(REVISION)
VPATH += ../src/external/fatfs
VPATH += ../src/external/segger
VPATH += ../src/peripherals/src
VPATH += ./peripherals

SRC =
SRC += am_devices_led.c
SRC += am_util_delay.c
SRC += am_util_stdio.c
SRC += am_util_string.c
SRC += ff.c
SRC += ffunicode.c
SRC += startup_gcc.c
SRC += SEGGER_RTT.c


SRC += logging.c
SRC += rtc.c
SRC += system.c

# Uncomment these lines to debug AmbiqSDK files
#SRC += $(foreach SOURCE,$(wildcard ../AmbiqSDK/mcu/$(PART)/hal/*.c),$(patsubst %.c,%.c,$(notdir $(SOURCE))))
#SRC += $(foreach SOURCE,$(wildcard ../AmbiqSDK/mcu/$(PART)/hal/mcu/*.c),$(patsubst %.c,%.c,$(notdir $(SOURCE))))

.PHONY: all program clean full logging rtc_set rtc system
.PRECIOUS: $(CONFIG)/%.axf
.SECONDEXPANSION:

all: setup
	$(error Make targets include: full logging rtc_set rtc system)

# TODO: Create separate targets for non-flashing versions
program:
ifneq ($(BUILT_BY), eclipse)
	printf "r\n" > $(CONFIG)/flash.jlink
	printf "loadfile $(CONFIG)/$(TARGET).bin $(FLASH_START)\nr\ng\nexit\n" >> $(CONFIG)/flash.jlink
	$(JLINK) $(JLINK_FLAGS) $(CONFIG)/flash.jlink
endif

full: TARGET = TestFull
full: SRC += main.c
full: SRC += ai.cpp
full: setup $(CONFIG) $(CONFIG)/main.o $(CONFIG)/ai.o $(CONFIG)/$$(TARGET).bin program

logging: TARGET = TestLogging
logging: SRC += test_logging.c
logging: $(CONFIG) $(CONFIG)/test_logging.o $(CONFIG)/$$(TARGET).bin program

rtc_set: TARGET = TestRTCSet
rtc_set: SRC += test_rtc.c
rtc_set: CFLAGS += -DSET_RTC
rtc_set: $(CONFIG) $(CONFIG)/test_rtc.o $(CONFIG)/$$(TARGET).bin program

rtc: TARGET = TestRTC
rtc: SRC += test_rtc.c
rtc: $(CONFIG) $(CONFIG)/test_rtc.o $(CONFIG)/$$(TARGET).bin program

system: TARGET = TestSystem
system: SRC += test_system.c
system: $(CONFIG) $(CONFIG)/test_system.o $(CONFIG)/$$(TARGET).bin program

CSRC = $(filter %.c,$(SRC))
ASRC = $(filter %.s,$(SRC))
CPPSRC = $(filter %.cpp,$(SRC))

OBJS = $(CSRC:%.c=$(CONFIG)/%.o)
OBJS += $(ASRC:%.s=$(CONFIG)/%.o)
OBJS += $(CPPSRC:%.cpp=$(CONFIG)/%.o)
OBJS += $(shell find ../src/ai/ -name '*.o')

DEPS = $(CSRC:%.c=$(CONFIG)/%.d)
DEPS += $(ASRC:%.s=$(CONFIG)/%.d)
DEPS += $(CPPSRC:%.cpp=$(CONFIG)/%.d)

LIBS  = ../AmbiqSDK/bsp/$(BSP)/gcc/bin/libam_bsp.a
LIBS += ../AmbiqSDK/mcu/$(PART)/hal/mcu/gcc/bin/libam_hal.a

CFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
CFLAGS+= -ffunction-sections -fdata-sections -fno-omit-frame-pointer
CFLAGS+= -MMD -MP -std=c99 -Wall -g -O0
CFLAGS+= $(DEFINES)
CFLAGS+= $(INCLUDES)

CPPFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
CPPFLAGS+= -ffunction-sections -fdata-sections -fomit-frame-pointer
CPPFLAGS+= -MMD -MP -Wall -Wno-alloc-size-larger-than -O3
CPPFLAGS+= -fno-exceptions -DNO_EXCEPTIONS
CPPFLAGS+= -fno-rtti
CPPFLAGS+= $(DEFINES)
CPPFLAGS+= $(INCLUDES)

LFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
LFLAGS+= -nostartfiles -static
LFLAGS+= -Wl,--gc-sections,--entry,Reset_Handler,-Map,$(CONFIG)/$(TARGET).map
LFLAGS+= -Wl,--start-group -lm -lc -lgcc -lnosys $(LIBS) -Wl,--end-group

CPFLAGS = -Obinary
ODFLAGS = -S

setup:
	+@cd ../src/ai && echo building ai components... && CCPP='$(CCPP) $(CPPFLAGS)' make && echo finished building ai components! && cd -

$(CONFIG):
	@mkdir -p $@

$(CONFIG)/%.o: %.c $(CONFIG)/%.d
	@echo " Compiling $<" ;\
	$(CC) -c $(CFLAGS) $< -o $@

$(CONFIG)/%.o: %.cpp $(CONFIG)/%.d
	@echo " Compiling $<" ;\
	$(CCPP) -c $(CPPFLAGS) $< -o $@

$(CONFIG)/%.o: %.s $(CONFIG)/%.d
	@echo " Assembling $<" ;\
	$(CC) -c $(CFLAGS) $< -o $@

$(CONFIG)/%.axf: $(OBJS) $(LIBS)
	@echo " Linking $@" ;\
	$(CC) -Wl,-T,$(LINKER_FILE) -o $@ $(OBJS) $(LFLAGS)

$(CONFIG)/%.bin: $(CONFIG)/%.axf
	@echo " Copying $@..." ;\
	$(CP) $(CPFLAGS) $< $@ ;\
	$(OD) $(ODFLAGS) $< > $(CONFIG)/$(TARGET).lst
	@$(SIZE) $(OBJS) $(LIBS) $(CONFIG)/$(TARGET).axf >$(CONFIG)/$(TARGET).size

clean:
	@echo "Cleaning..." ;\
	$(RM) -rf $(CONFIG) ;\
	cd ../src/ai && make clean && cd -

$(CONFIG)/%.d: ;

# Include JTag flashing Makefile
include ../Jtag.mk

# Automatically include any generated dependencies
-include $(DEPS)

endif
