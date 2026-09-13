import { Slot } from "@radix-ui/react-slot";
import { cva, type VariantProps } from "class-variance-authority";
import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";
import type { ComponentProps } from "react";

const variants = cva(
  "ui-button inline-flex items-center justify-center gap-2 disabled:opacity-50",
  {
    variants: {
      variant: {
        default: "primary-command",
        outline: "",
        ghost: "ui-button-ghost",
      },
      size: { default: "", icon: "icon-button" },
    },
    defaultVariants: { variant: "default", size: "default" },
  },
);
export function Button({
  asChild,
  variant,
  size,
  className,
  ...props
}: ComponentProps<"button"> &
  VariantProps<typeof variants> & { asChild?: boolean }) {
  const Component = asChild ? Slot : "button";
  return (
    <Component
      data-slot="button"
      className={twMerge(clsx(variants({ variant, size }), className))}
      {...props}
    />
  );
}
