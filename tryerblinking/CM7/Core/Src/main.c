#include "stm32h7xx_hal.h"

int main(void)
{
    // Initialize HAL library
    HAL_Init();

    // Enable GPIOC clock
    __HAL_RCC_GPIOC_CLK_ENABLE();

    // Configure PC7 as output
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOC, &GPIO_InitStruct);

    // Main loop
    while (1)
    {
        HAL_GPIO_TogglePin(GPIOC, GPIO_PIN_7); // Toggle LED
        HAL_Delay(500);                        // 500 ms delay
    }
}
