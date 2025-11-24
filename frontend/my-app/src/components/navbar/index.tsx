import React from "react";
import { Montserrat } from "next/font/google"
import Link from "next/link";

const montserratFont = Montserrat({
    subsets :["latin"],
    weight : "400",
})
// test github
export default function Navbar (){
    return (
        <div className={`text-black bg-transparent ${montserratFont.className}`}>
            <div className="p-10 pt-8 pb-5">
                <div className="">
                    <div className="flex flex-row justify-between">
                        {/* logo */}
                        <div>
                            <Link href="/">
                                <img src="/logo/sbd1.svg" className="w-30 h-auto "alt="" />
                            </Link>
                        </div>

                        <div className="flex justify-center items-center gap-x-10">
                            <span>
                                <Link href="/Introduction">
                                    Introduction
                                </Link>
                            </span>
                            <span>
                                <Link href="/About-Us">
                                    About Us
                                </Link>
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}