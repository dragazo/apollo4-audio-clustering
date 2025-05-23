PROJECT := Apollo4StarterCode
TARGET := Apollo4StarterCode
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

$(info Building for Revision $(REVISION):)
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
DEFINES += -DAM_PACKAGE_BGA
DEFINES += -Dgcc

LINKER_FILE := ./AmbiqSDK/bsp/$(BSP)/linker/a3em.ld
STARTUP_FILE := ./AmbiqSDK/bsp/$(BSP)/linker/startup_gcc.c

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

INCLUDES  = -IAmbiqSDK/bsp/$(BSP)
INCLUDES += -IAmbiqSDK/mcu/$(PART)
INCLUDES += -IAmbiqSDK/mcu/$(PART)/hal
INCLUDES += -IAmbiqSDK/mcu/$(PART)/hal/mcu
INCLUDES += -IAmbiqSDK/CMSIS/AmbiqMicro/Include
INCLUDES += -IAmbiqSDK/CMSIS/ARM/Include
INCLUDES += -IAmbiqSDK/devices
INCLUDES += -IAmbiqSDK/utils
INCLUDES += -Isrc/app
INCLUDES += -Isrc/boards
INCLUDES += -Isrc/boards/rev$(REVISION)
INCLUDES += -Isrc/external/fatfs
INCLUDES += -Isrc/peripherals/include

VPATH  = AmbiqSDK/bsp/$(BSP)/linker
VPATH += AmbiqSDK/devices
VPATH += AmbiqSDK/utils
VPATH += src/app
VPATH += src/boards
VPATH += src/boards/rev$(REVISION)
VPATH += src/external/fatfs
VPATH += src/peripherals/src

SRC =
SRC += am_devices_led.c
SRC += am_util_delay.c
SRC += am_util_stdio.c
SRC += am_util_string.c
SRC += ff.c
SRC += ffunicode.c
SRC += startup_gcc.c

SRC += logging.c
SRC += rtc.c
SRC += system.c
SRC += main.c
SRC += ai.cpp

CSRC = $(filter %.c,$(SRC))
ASRC = $(filter %.s,$(SRC))
CPPSRC = $(filter %.cpp,$(SRC))

OBJS = $(CSRC:%.c=$(CONFIG)/%.o)
OBJS += $(ASRC:%.s=$(CONFIG)/%.o)
OBJS += $(CPPSRC:%.cpp=$(CONFIG)/%.o)
OBJS += $(shell find src/ai/ -name '*.o' | grep -Fxv src/ai/build/test.o)

DEPS  = $(CSRC:%.c=$(CONFIG)/%.d)
DEPS += $(ASRC:%.s=$(CONFIG)/%.d)
DEPS += $(CPPSRC:%.cpp=$(CONFIG)/%.d)

LIBS = AmbiqSDK/bsp/$(BSP)/gcc/bin/libam_bsp.a
LIBS += AmbiqSDK/mcu/$(PART)/hal/mcu/gcc/bin/libam_hal.a

CFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
CFLAGS += -ffunction-sections -fdata-sections -fomit-frame-pointer
CFLAGS += -MMD -MP -std=c99 -Wall -O3
CFLAGS += $(DEFINES)
CFLAGS += $(INCLUDES)

CPPFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
CPPFLAGS += -ffunction-sections -fdata-sections -fomit-frame-pointer
CPPFLAGS += -MMD -MP -Wall -Wno-alloc-size-larger-than -O3
CPPFLAGS += -fno-exceptions -DNO_EXCEPTIONS
CPPFLAGS += -fno-rtti
CPPFLAGS += $(DEFINES)
CPPFLAGS += $(INCLUDES)

LFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
LFLAGS += -nostartfiles -static
LFLAGS += -Wl,--gc-sections,--entry,Reset_Handler,-Map,$(CONFIG)/$(TARGET).map
LFLAGS += -Wl,--start-group -lm -lc -lgcc -lnosys $(LIBS) -Wl,--end-group

CPFLAGS = -Obinary
ODFLAGS = -S

#### Rules ####

.PHONYY: setup all

setup:
	+@cd src/ai && echo building ai components... && CCPP='$(CCPP) $(CPPFLAGS)' make && echo finished building ai components! && cd -

all: setup directories $(CONFIG)/$(TARGET).bin

directories: $(CONFIG)

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

$(CONFIG)/$(TARGET).axf: $(OBJS) $(LIBS)
	@echo " Linking $@" ;\
	$(CC) -Wl,-T,$(LINKER_FILE) -o $@ $(OBJS) $(LFLAGS)

$(CONFIG)/$(TARGET).bin: $(CONFIG)/$(TARGET).axf
	@echo " Copying $@..." ;\
	$(CP) $(CPFLAGS) $< $@ ;\
	$(OD) $(ODFLAGS) $< > $(CONFIG)/$(TARGET).lst
	@$(SIZE) $(OBJS) $(LIBS) $(CONFIG)/$(TARGET).axf >$(CONFIG)/$(TARGET).size

clean:
	@echo "Cleaning..." ;\
	$(RM) -rf $(CONFIG) ;\
	cd src/ai && make clean && cd -

$(CONFIG)/%.d: ;

# Include JTag flashing Makefile
include Jtag.mk

# Automatically include any generated dependencies
-include $(DEPS)

endif
.PHONY: all clean directories