"use client";

import React, { useState } from "react";
import { Montserrat } from "next/font/google";
import Link from "next/link";
import Image from "next/image";

const montserratFont = Montserrat({
    subsets: ["latin"],
    weight: ["400", "600"],
});

export default function Navbar() {
    const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

    return (
        <nav className={`text-black bg-white/95 backdrop-blur-sm border-b border-gray-100 ${montserratFont.className}`}>
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex justify-between items-center h-16">
                    {/* Logo */}
                    <Link href="/" className="shrink-0">
                        <Image 
                            src="/logo/sbd1.svg" 
                            alt="SBD Logo" 
                            width={120} 
                            height={44}
                            className="h-10 w-auto"
                            priority
                        />
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center gap-8">
                        <Link 
                            href="/About-Us"
                            className="text-gray-700 hover:text-blue-600 transition-colors font-medium"
                        >
                            About Us
                        </Link>
                    </div>

                    {/* Mobile menu button */}
                    <button
                        onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
                        className="md:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
                        aria-label="Toggle menu"
                    >
                        <svg
                            className="w-6 h-6"
                            fill="none"
                            stroke="currentColor"
                            viewBox="0 0 24 24"
                        >
                            {mobileMenuOpen ? (
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M6 18L18 6M6 6l12 12"
                                />
                            ) : (
                                <path
                                    strokeLinecap="round"
                                    strokeLinejoin="round"
                                    strokeWidth={2}
                                    d="M4 6h16M4 12h16M4 18h16"
                                />
                            )}
                        </svg>
                    </button>
                </div>
            </div>

            {/* Mobile Menu */}
            {mobileMenuOpen && (
                <div className="md:hidden border-t border-gray-100 bg-white">
                    <div className="px-4 py-3 space-y-2">
                        <Link
                            href="/About-Us"
                            className="block px-4 py-2 text-gray-700 hover:bg-gray-50 rounded-lg transition-colors"
                            onClick={() => setMobileMenuOpen(false)}
                        >
                            About Us
                        </Link>
                    </div>
                </div>
            )}
        </nav>
    );
}