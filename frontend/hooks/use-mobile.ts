import { useEffect, useState } from "react"
import { Dimensions } from 'react-native'

const MOBILE_BREAKPOINT = 768

export function useIsMobile() {
  const [isMobile, setIsMobile] = useState<boolean | undefined>(undefined)

  useEffect(() => {
    const subscription = Dimensions.addEventListener('change', ({ window }) => {
      setIsMobile(window.width < MOBILE_BREAKPOINT)
    })

    // Set initial value
    const { width } = Dimensions.get('window')
    setIsMobile(width < MOBILE_BREAKPOINT)

    return () => subscription?.remove()
  }, [])

  return !!isMobile
}
